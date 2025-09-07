use std::{
    fs::File,
    io::{self, Result},
    path::Path,
};

use memmap2::Mmap;

pub struct Config {
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub vocab_size: i32,
    pub seq_len: i32,
}

impl Config {
    /// used to parse checkpoint header from a raw little-endian byte buffer into Config struct
    fn from_le_bytes(hdr: &[u8]) -> Result<Self> {
        let need = size_of::<Config>();
        if hdr.len() < need {
            println!("header too small: have {}, need {}", hdr.len(), need)
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
    pub _mmap: Mmap,
    // pub state: RunState, // buffers for the "wave" of activations in the forward pass
    // // some more state needed to properly clean up the memory mapping (sigh)
    // pub fd: i32,        // file descriptor for memory mapping
    // pub data: Vec<f32>, // memory mapped data pointer
    // pub file_size: i32, // size of the checkpoint file in bytes
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
    fn take<'a>(buffer: &'a [f32], offset: &mut usize, size: i32) -> &'a [f32] {
        let start = *offset;
        let end = start + size as usize;
        // we guarantee that `buffer` is large enough.
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
    offset += rope_skip as usize * 2; // skip both halves

    w.wcls = if shared_weights != 0 {
        w.token_embedding_table
    } else {
        take(ptr, &mut offset, p.vocab_size * p.dim)
    };
}

// pub fn read_checkpoint<'a>(
//     checkpoint: &Path,
//     weights_out: &'a mut TransformerWeights<'a>,
// ) -> io::Result<Config> {
//     // get file from checkpoint path
//     let mut f = File::open(checkpoint)?;

//     // get file size
//     let file_size = f.seek(SeekFrom::End(0))?;

//     // go to the byte at the 0th offset position aka the beginning of the file
//     f.seek(SeekFrom::Start(0))?;

//     // initialize slice the size of the header
//     let mut hdr = vec![0u8; size_of::<Config>()];

//     // reads in the exact number of bytes to fill in the hdr buffer
//     f.read_exact(&mut hdr)?;

//     //assigns the bytes from the header to the config struct fields
//     let mut cfg =
//         Config::from_le_bytes(&hdr).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

//     // negative vocab size is hacky way of signaling unshared weights. bit yikes.
//     let shared_weights = cfg.vocab_size > 0;
//     if cfg.vocab_size < 0 {
//         cfg.vocab_size = cfg.vocab_size * -1
//     } else {
//         cfg.vocab_size = cfg.vocab_size
//     }

//     // calculate how many bytes of weights after the header are in the file by subtracting the size of the header from the file size
//     let payload_bytes = (file_size as usize).saturating_sub(size_of::<Config>());
//     if payload_bytes % 4 != 0 {
//         return Err(io::Error::new(
//             io::ErrorKind::InvalidData,
//             "weights length not multiple of 4",
//         ));
//     }

//     // create a slice that size of the weight bytes
//     let mut raw = vec![0u8; payload_bytes];

//     // fills the slice with the bytes from the file
//     f.read_exact(&mut raw)?;

//     // creats a vector to hold the f32 weights
//     let mut floats = Vec::<f32>::with_capacity(payload_bytes / 4);

//     // converts the bytes into weights
//     for chunk in raw.chunks_exact(4) {
//         floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
//     }

//     // the floats need to live for the lifetime of the progra otherwise they will be dropped since floats is a local variable - this is because the memory_map_weights take in slices which just borrow data (if they were vectors they would own the data and we wouldn't see this issue) but im trying to keep it similar to teh C implementation
//     // so we use leak to keep the storage alive and fix the lifetime issue
//     let slice: &'static [f32] = Box::leak(floats.into_boxed_slice());
//     memory_map_weights(weights_out, &cfg, slice, shared_weights as i32);

//     Ok(cfg)
// }

// fn build_transformer<'a>(t: &'a mut Transformer<'a>, checkpoint_path: &str) -> io::Result<()> {
//     let path = Path::new(checkpoint_path);
//     let _cfg = read_checkpoint(path, &mut t.weights)?;
//     Ok(())
// }

pub fn read_checkpoint(checkpoint: &Path) -> io::Result<(Config, Mmap, bool)> {
    // Open file and create memory map
    let file = File::open(checkpoint)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Validate minimum size
    if mmap.len() < size_of::<Config>() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "File too small for config header",
        ));
    }

    // Read config from the beginning of the mmap
    let config_bytes = &mmap[0..size_of::<Config>()];
    let mut cfg = Config::from_le_bytes(config_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let shared_weights = cfg.vocab_size > 0;
    if cfg.vocab_size < 0 {
        cfg.vocab_size = cfg.vocab_size * -1
    } else {
        cfg.vocab_size = cfg.vocab_size
    }

    // Validate payload size
    let payload_bytes = mmap.len().saturating_sub(size_of::<Config>());
    if payload_bytes % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "weights length not multiple of 4",
        ));
    }

    Ok((cfg, mmap, shared_weights))
}

fn build_transformer(checkpoint_path: &str, t: &mut Transformer) {
    let path = Path::new(checkpoint_path);
    let (cfg, mmap, sw) = read_checkpoint(path).unwrap();

    // Get the weights data from the mmap
    let weights_bytes = &mmap[size_of::<Config>()..];

    // Convert bytes to f32 slice (this is safe because we validated alignment)
    let weights_data: &[f32] = unsafe {
        std::slice::from_raw_parts(
            weights_bytes.as_ptr() as *const f32,
            weights_bytes.len() / 4,
        )
    };

    memory_map_weights(&mut t.weights, &cfg, weights_data, sw as i32);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
    let mut ss = 0.0f32;

    for j in (0..size) {
        ss += x[j] * x[j]
    }

    ss /= size as f32;
    ss += 1e-5f32;
    ss = 1.0f32 / ss.sqrt();

    for j in 0..size {
        o[j] = weight[j] * (ss * x[j]);
    }
}

fn softmax(x: &mut [f32], size: usize) {
    let mut max_val = x[0];

    for i in 0..size {
        if (x[i] > max_val) {
            max_val = x[i]
        }
    }

    let mut sum = 0.0f32;
    for i in 0..size {
        x[i] = (x[i] - max_val).exp();
        sum += x[1]
    }

    for i in 0..size {
        x[i] /= sum
    }
}

fn main() {
    println!("Hello, world!");
}
