// a port of karpathy's llama2.c in pure rust

use std::{
    fs::File,
    io::{self, Read, Result, Seek, SeekFrom},
    path::Path,
};

pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: i32,
    pub seq_len: usize,
}

// header is parsed into this. we could make the config have i32 fields but it messes up the memory map weights function which deals with usizes, maybe come back later an dupdate it adn then we can get rid of this struct
#[derive(Debug, Clone, Copy)]
struct DiskConfig {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32, // may be NEGATIVE in file to signal "unshared"
    seq_len: i32,
}

impl DiskConfig {
    /// Parse from a raw little-endian byte buffer
    fn from_le_bytes(hdr: &[u8]) -> Result<Self> {
        let need = size_of::<DiskConfig>();
        if hdr.len() < need {
            println!("header is too small")
        }
        let mut off = 0usize;
        let mut next_i32 = || {
            let b = [hdr[off], hdr[off + 1], hdr[off + 2], hdr[off + 3]];
            off += 4;
            i32::from_le_bytes(b)
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

pub fn read_checkpoint<'a>(
    checkpoint: &Path,
    weights_out: &'a mut TransformerWeights<'a>,
) -> Result<Config> {
    let mut f = File::open(checkpoint)?;

    // get file size
    let file_size = f.seek(SeekFrom::End(0))?;
    f.seek(SeekFrom::Start(0))?;

    // slice the size of the header
    let mut hdr = vec![0u8; size_of::<DiskConfig>()];
    f.read_exact(&mut hdr)?;

    // can do this in one step if we change the type of the config fields in the struct from usize -> i32
    let disk = DiskConfig::from_le_bytes(&hdr)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let (mut cfg, shared) =
        to_runtime_config(disk).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let shared_weights = cfg.vocab_size > 0;
    cfg.vocab_size = cfg.vocab_size.abs();

    // read payload
    let payload_bytes = (file_size as usize).saturating_sub(size_of::<DiskConfig>());
    if payload_bytes % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "weights length not multiple of 4",
        ));
    }

    let mut raw = vec![0u8; payload_bytes];
    f.read_exact(&mut raw)?;

    // bytes -> f32 (safe conversion)
    let mut floats = Vec::<f32>::with_capacity(payload_bytes / 4);
    for chunk in raw.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    // keep storage alive (own it somewhere; demo uses a leak to keep lifetimes simple)
    let boxed: Box<[f32]> = floats.into_boxed_slice();
    let all_f32: &'static [f32] = Box::leak(boxed);

    memory_map_weights(weights_out, &cfg, all_f32, shared_weights as i32);

    Ok(cfg)
}

fn to_runtime_config(disk: DiskConfig) -> Result<(Config, bool)> {
    // negative vocab means "unshared" in the legacy format; take absolute for value
    let shared_weights = disk.vocab_size > 0;
    let vocab_abs = disk.vocab_size.abs();

    // Ensure all fields are non-negative and fit into usize
    fn cast(n: i32, name: &str) -> Result<usize> {
        if n < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "field must be >= 0",
            ));
        }
        Ok(n as usize)
    }

    let cfg = Config {
        dim: cast(disk.dim, "dim")?,
        hidden_dim: cast(disk.hidden_dim, "hidden_dim")?,
        n_layers: cast(disk.n_layers, "n_layers")?,
        n_heads: cast(disk.n_heads, "n_heads")?,
        n_kv_heads: cast(disk.n_kv_heads, "n_kv_heads")?,
        vocab_size: vocab_abs,
        seq_len: cast(disk.seq_len, "seq_len")?,
    };
    Ok((cfg, shared_weights))
}

fn main() {
    println!("Hello, world!");
}
