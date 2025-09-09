use memmap2::Mmap;
use rayon::prelude::*;
use std::{
    fs::File,
    io::{self, Result},
    path::Path,
};

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

pub struct RunState<'a> {
    // current wave of activations
    pub x: &'a [f32],      // activation at current time stamp (dim,)
    pub xb: &'a [f32],     // same, but inside a residual branch (dim,)
    pub xb2: &'a [f32],    // an additional buffer just for convenience (dim,)
    pub hb: &'a [f32],     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: &'a [f32],    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: &'a [f32],      // query (dim,)
    pub k: &'a [f32],      // key (dim,)
    pub v: &'a [f32],      // value (dim,)
    pub att: &'a [f32],    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: &'a [f32], // output logits
    // kv cache
    pub key_cache: &'a [f32],   // (layer, seq_len, dim)
    pub value_cache: &'a [f32], // (layer, seq_len, dim)
}

pub struct Transformer<'a> {
    pub config: Config, // the hyperparameters of the architecture (the blueprint)
    pub weights: TransformerWeights<'a>, // the weights of the model
    pub _mmap: Mmap,
    pub state: RunState<'a>, // buffers for the "wave" of activations in the forward pass
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

fn matmul(xout: &mut [f32], x: &mut [f32], w: &mut [f32], n: usize, d: usize) {
    xout.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let mut val = 0.0f32;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        *out_val = val;
    });
}

fn forward<'a>(transformer: &'a mut Transformer, token: usize, pos: usize) -> &'a [f32] {
    let p = &transformer.config;
    let w = &transformer.weights;
    let s = &mut transformer.state;

    // calc dims for forward pass
    let dim = p.dim as usize;
    let kv_dim = ((p.dim * p.n_kv_heads) / p.n_heads) as usize;
    let kv_mul = (p.n_heads / p.n_kv_heads) as usize;
    let hidden_dim = p.hidden_dim as usize;
    let head_size = (p.dim / p.n_heads) as usize;

    // copy the token embedding into x
    let start = token * dim;
    let end = start + dim;
    let content_row = &w.token_embedding_table[start..end];
    s.x[..dim].copy_from_slice(&content_row[..dim]);

    // forward all the layers
    for l in 0..p.n_layers as usize {
        // attention rmsnorm
        let rms_offset = l * dim;
        let rms_weights = &w.rms_att_weight[rms_offset..rms_offset + dim];
        rmsnorm(&mut s.xb, &s.x, rms_weights, p.dim as usize);

        // key and value point to the kv cache
        let loff = l * p.seq_len as usize * kv_dim; // kv cache layer offset for convenience
        let k_start = loff + pos * kv_dim;
        let v_start = loff + pos * kv_dim;

        // qkv matmuls for this position
        let wq_offset = l * dim * dim;
        let wk_offset = l * dim * kv_dim;
        let wv_offset = l * dim * kv_dim;

        let mut wq_slice = &w.wq[wq_offset..wq_offset + dim * dim];
        let mut wk_slice = &w.wk[wk_offset..wk_offset + dim * kv_dim];
        let mut wv_slice = &w.wv[wv_offset..wv_offset + dim * kv_dim];

        matmul(&mut s.q, &mut s.xb, &mut wq_slice, dim, dim);
        matmul(
            &mut s.key_cache[k_start..k_start + kv_dim],
            &mut s.xb,
            &mut wq_slice,
            dim,
            kv_dim,
        );
        matmul(
            &mut s.value_cache[v_start..v_start + kv_dim],
            &mut s.xb,
            &mut wq_slice,
            dim,
            kv_dim,
        );

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        let mut i = 0;
        while i < dim {
            let head_dim = (i % head_size) as f32;
            let freq = 1.0f32 / 10000.0f32.powf(head_dim / head_size as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < kv_dim { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only

            for v in 0..rotn {
                let vec = if v == 0 {
                    &mut s.q
                } else {
                    &mut s.key_cache[k_start..k_start + kv_dim]
                }; // the vector to rotate (query or key)
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }

            i += 2;
        }

        // multihead attention. iterate over all heads
        for h in 0..p.n_heads as usize {
            // get the query vector for this head
            let q_start = h * head_size;
            let q_slice = &s.q[q_start..q_start + head_size];

            // attention scores for this head
            let att_start = h * p.seq_len as usize;
            let att_slice = &mut s.att[att_start..att_start + p.seq_len as usize];

            // iterate over all timesteps, including the current one
            for t in 0..=pos {
                // get the key vector for this head and at this timestep
                let k_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                let k_slice = &s.key_cache[k_offset..k_offset + head_size];

                // calculate the attention score as the dot product of q and k
                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q_slice[i] * k_slice[i];
                }
                score /= (head_size as f32).sqrt();

                // save the score to the attention buffer
                att_slice[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut att_slice[0..=pos], pos + 1);

            // weighted sum of the values, store back into xb
            let xb_start = h * head_size;
            let xb_slice = &mut s.xb[xb_start..xb_start + head_size];

            // zero out the slice
            for val in xb_slice.iter_mut() {
                *val = 0.0;
            }

            for t in 0..=pos {
                // get the value vector for this head and at this timestep
                let v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                let v_slice = &s.value_cache[v_offset..v_offset + head_size];

                // get the attention weight for this timestep
                let a = att_slice[t];

                // accumulate the weighted value into xb
                for i in 0..head_size {
                    xb_slice[i] += a * v_slice[i];
                }
            }
        }

        // final matmul to get the output of the attention
        let wo_offset = l * dim * dim;
        let wo_slice = &w.wo[wo_offset..wo_offset + dim * dim];
        matmul(&mut s.xb2, &s.xb, wo_slice, dim, dim);

        // residual connection back into x
        for i in 0..dim {
            s.x[i] += s.xb2[i];
        }

        // ffn rmsnorm
        let rms_ffn_offset = l * dim;
        let rms_ffn_weights = &w.rms_ffn_weight[rms_ffn_offset..rms_ffn_offset + dim];
        rmsnorm(&mut s.xb, &s.x, rms_ffn_weights, p.dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        let w1_offset = l * dim * hidden_dim;
        let w3_offset = l * dim * hidden_dim;
        let w1_slice = &w.w1[w1_offset..w1_offset + dim * hidden_dim];
        let w3_slice = &w.w3[w3_offset..w3_offset + dim * hidden_dim];

        matmul(&mut s.hb, &s.xb, w1_slice, dim, hidden_dim);
        matmul(&mut s.hb2, &s.xb, w3_slice, dim, hidden_dim);

        // SwiGLU non-linearity
        for i in 0..hidden_dim {
            let mut val = s.hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= 1.0f32 / (1.0f32 + (-val).exp());
            // elementwise multiply with w3(x)
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        // final matmul to get the output of the ffn
        let w2_offset = l * dim * hidden_dim;
        let w2_slice = &w.w2[w2_offset..w2_offset + dim * hidden_dim];
        matmul(&mut s.xb, &s.hb, w2_slice, hidden_dim, dim);

        // residual connection
        for i in 0..dim {
            s.x[i] += s.xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(&mut s.x, &s.x.clone(), &w.rms_final_weight, p.dim);

    // classifier into logits
    matmul(
        &mut s.logits,
        &s.x,
        &w.wcls,
        p.dim as usize,
        p.vocab_size as usize,
    );

    &s.logits
}

fn main() {
    println!("Hello, world!");
}
