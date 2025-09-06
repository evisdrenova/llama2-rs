// a port of karpathy's llama2.c in pure rust

use std::{
    error,
    fs::{File, OpenOptions},
    io::{self, Read, Result},
    path::Path,
};

pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

impl Config {
    fn parse_le_bytes(header: &[u8]) -> Result<Self, String> {
        let need = size_of::<Config>();
        if header.len() < need {
            return Err(format!(
                "header too small: have {}, need {}",
                hdr.len(),
                need
            ));
        }

        let mut off = 0usize;
        let mut next_i32 = || {
            let bytes = [
                header[offset],
                header[offset + 1],
                header[offset + 2],
                header[offset + 3],
            ];
            offset += 4;
            i32::from_le_bytes(bytes)
        };

        Ok(Self {
            dim: next_i32(),
            hidden_dim: next_i32(),
            n_layers: next_i32(),
            n_heads: next_i32(),
            n_kv_heads: next_i32(),
            vocab_size: next_i32(),
            seq_len: next_i32(),
        })
    }
}

// the original c impleementation has this as float pointers but we're goin to change them to slices of f32s so we can memory map them correctly. more on this below.
pub struct TransformerWeights<'a> {
    // token embedding table
    pub token_embedding_table: &'a [f32],
    // weights for rmsnorms
    pub rms_att_weight: &'a [f32], // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: &'a [f32], // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: &'a [f32], // (layer, dim, n_heads * head_size)
    pub wk: &'a [f32], // (layer, dim, n_kv_heads * head_size)
    pub wv: &'a [f32], // (layer, dim, n_kv_heads * head_size)
    pub wo: &'a [f32], // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: &'a [f32], // (layer, hidden_dim, dim)
    pub w2: &'a [f32], // (layer, dim, hidden_dim)
    pub w3: &'a [f32], // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: &'a [f32], // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: &'a [f32],
}

pub struct RunState {
    // current wave of activations
    pub x: Vec<f32>,      // activation at current time stamp (dim,)
    pub xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,      // query (dim,)
    pub k: Vec<f32>,      // key (dim,)
    pub v: Vec<f32>,      // value (dim,)
    pub att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<f32>, // output logits
    // kv cache
    pub key_cache: Vec<f32>,   // (layer, seq_len, dim)
    pub value_cache: Vec<f32>, // (layer, seq_len, dim)
}

pub struct Transformer<'a> {
    pub config: Config, // the hyperparameters of the architecture (the blueprint)
    pub weights: TransformerWeights<'a>, // the weights of the model
    pub state: RunState, // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    pub fd: i32,        // file descriptor for memory mapping
    pub data: Vec<f32>, // memory mapped data pointer
    pub file_size: i32, // size of the checkpoint file in bytes
}

pub fn memory_map_weights<'a>(
    w: &mut TransformerWeights<'a>,
    p: &Config,
    ptr: &'a [f32],
    shared_weights: i32,
) {
    let head_size = p.dim / p.n_layers;
    let n_layers = p.n_layers;

    let mut offset = 0;

    //take a slice of floats starting at offset and advance the offset so that the next call starts after it. this is functionally equivaletn to using a raw pointer to hop around the weights array
    //No data are moved, the slice is just a new pair (pointer, length)
    fn take<'a>(buffer: &'a [f32], offset: &mut usize, size: usize) -> &'a [f32] {
        let start = *offset;
        let end = start + size;
        // safety: the caller guarantees that `buffer` is large enough.
        let slice = &buffer[start..end];
        *offset = end;
        slice
    }

    w.token_embedding_table = take(ptr, &mut offset, p.vocab_size * p.dim);
    w.rms_att_weight = take(ptr, &mut offset, n_layers * p.dim);
    w.wq = take(ptr, &mut offset, n_layers * p.dim * (p.n_heads * head_size));
    w.wk = take(
        ptr,
        &mut offset,
        n_layers * p.dim * (p.n_kv_heads * head_size),
    );
    w.wv = take(
        ptr,
        &mut offset,
        n_layers * p.dim * (p.n_kv_heads * head_size),
    );
    w.wo = take(ptr, &mut offset, n_layers * (p.n_heads * head_size) * p.dim);
    w.rms_ffn_weight = take(ptr, &mut offset, n_layers * p.dim);
    w.w1 = take(ptr, &mut offset, n_layers * p.dim * p.hidden_dim);
    w.w2 = take(ptr, &mut offset, n_layers * p.hidden_dim * p.dim);
    w.w3 = take(ptr, &mut offset, n_layers * p.dim * p.hidden_dim);
    w.rms_final_weight = take(ptr, &mut offset, p.dim);

    // Skip the two RoPE tables (freq_cis_real/imag)
    let rope_skip = p.seq_len * head_size / 2; // each half
    offset += rope_skip * 2; // skip both halves

    w.wcls = if shared_weights != 0 {
        w.token_embedding_table
    } else {
        take(ptr, &mut offset, p.vocab_size * p.dim)
    };
}

fn read_checkpoint<'a>(
    checkpoint: &str,
    config: &mut Config,
    weights: &mut TransformerWeights,
    fd: usize,
    data: &'a [f32],
    file_size: usize,
) -> Result<(File, Mmap), Box<dyn std::error::Error>> {
    let mut file = File::open(checkpoint)?;

    // Read config header using safe deserialization
    let config_bytes = read_exact_bytes(&mut file, size_of::<Config>())?;
    let parsed_config = parse_config_from_bytes(&config_bytes)?;
}

fn main() {
    println!("Hello, world!");
}
