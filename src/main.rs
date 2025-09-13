use memmap2::Mmap;
use rayon::prelude::*;
use std::io::BufRead;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    cmp::Ordering,
    fs::File,
    io::{self, BufReader, Read, Result, Write},
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
    pub x: &'a mut [f32],      // activation at current time stamp (dim,)
    pub xb: &'a mut [f32],     // same, but inside a residual branch (dim,)
    pub xb2: &'a mut [f32],    // an additional buffer just for convenience (dim,)
    pub hb: &'a mut [f32],     // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: &'a mut [f32],    // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: &'a mut [f32],      // query (dim,)
    pub k: &'a mut [f32],      // key (dim,)
    pub v: &'a [f32],          // value (dim,)
    pub att: &'a mut [f32],    // buffer for scores/attention values (n_heads, seq_len)
    pub logits: &'a mut [f32], // output logits
    // kv cache
    pub key_cache: &'a mut [f32],   // (layer, seq_len, dim)
    pub value_cache: &'a mut [f32], // (layer, seq_len, dim)
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
    assert!(size <= x.len(), "softmax size > slice len");
    if size == 0 {
        return;
    }

    let slice = &mut x[..size];

    // 1) subtract max for numerical stability
    let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // 2) exponentiate and sum
    let mut sum = 0.0f32;
    for v in slice.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // 3) normalize (fallback to uniform if sum underflows)
    if sum > 0.0 {
        for v in slice.iter_mut() {
            *v /= sum;
        }
    } else {
        let uniform = 1.0 / (size as f32);
        for v in slice.iter_mut() {
            *v = uniform;
        }
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
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

            // How many positions are valid for this step?
            let cur_len = (pos + 1).min(p.seq_len as usize);

            // iterate over all timesteps, including the current one
            for t in 0..cur_len {
                let k_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                let k_slice = &s.key_cache[k_offset..k_offset + head_size];

                let mut score = 0.0f32;
                for i in 0..head_size {
                    score += q_slice[i] * k_slice[i];
                }
                score /= (head_size as f32).sqrt();

                att_slice[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut att_slice[..cur_len], cur_len);

            // weighted sum of the values, store back into xb
            let xb_start = h * head_size;
            let xb_slice = &mut s.xb[xb_start..xb_start + head_size];
            for val in xb_slice.iter_mut() {
                *val = 0.0;
            }

            for t in 0..cur_len {
                let v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                let v_slice = &s.value_cache[v_offset..v_offset + head_size];
                let a = att_slice[t];
                for i in 0..head_size {
                    xb_slice[i] += a * v_slice[i];
                }
            }
        }

        // final matmul to get the output of the attention
        let wo_offset = l * dim * dim;
        let wo_slice = &w.wo[wo_offset..wo_offset + dim * dim];
        matmul(&mut s.xb2, &mut s.xb, wo_slice, dim, dim);

        // residual connection back into x
        for i in 0..dim {
            s.x[i] += s.xb2[i];
        }

        // ffn rmsnorm
        let rms_ffn_offset = l * dim;
        let rms_ffn_weights = &w.rms_ffn_weight[rms_ffn_offset..rms_ffn_offset + dim];
        rmsnorm(&mut s.xb, &s.x, rms_ffn_weights, p.dim as usize);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        let w1_offset = l * dim * hidden_dim;
        let w3_offset = l * dim * hidden_dim;
        let w1_slice = &w.w1[w1_offset..w1_offset + dim * hidden_dim];
        let w3_slice = &w.w3[w3_offset..w3_offset + dim * hidden_dim];

        matmul(&mut s.hb, &mut s.xb, &w1_slice, dim, hidden_dim);
        matmul(&mut s.hb2, &mut s.xb, &w3_slice, dim, hidden_dim);

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
        let mut w2_slice = &w.w2[w2_offset..w2_offset + dim * hidden_dim];
        matmul(&mut s.xb, &mut s.hb, w2_slice, hidden_dim, dim);
        matmul(&mut s.xb, &mut s.hb, w2_slice, hidden_dim, dim);

        // residual connection
        for i in 0..dim {
            s.x[i] += s.xb[i];
        }
    }

    let x_copy = s.x.to_vec();
    rmsnorm(&mut s.x, &x_copy, &w.rms_final_weight, p.dim as usize);

    // classifier into logits
    matmul(
        &mut s.logits,
        s.x,
        w.wcls,
        p.dim as usize,
        p.vocab_size as usize,
    );

    &s.logits
}

pub struct TokenIndex {
    str: String,
    id: usize,
}

pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    sorted_vocab: Option<Vec<TokenIndex>>,
    vocab_size: i32,
    max_token_length: u32,
    byte_pieces: [u8; 512],
}

fn compare_tokens(a: &TokenIndex, b: &TokenIndex) -> Ordering {
    a.str.cmp(&b.str)
}

fn build_tokenizer<'a>(t: &mut Tokenizer, tokenizer_path: &'a str, vocab_size: i32) {
    let mut tokenizer = Tokenizer {
        vocab_size,
        vocab: Vec::with_capacity(vocab_size as usize),
        vocab_scores: Vec::with_capacity(vocab_size as usize),
        sorted_vocab: None,
        max_token_length: 0,
        byte_pieces: [0u8; 512],
    };

    for i in 0..256 {
        tokenizer.byte_pieces[i * 2] = i as u8;
        tokenizer.byte_pieces[i * 2 + 1] = 0u8;
    }

    let file = File::open(tokenizer_path).unwrap();
    let mut reader = BufReader::new(file);
    let mut len: i32;

    for i in 0..vocab_size {
        // Read vocab score
        let mut score_bytes = [0u8; size_of::<f32>()];
        reader
            .read_exact(&mut score_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "failed read"))
            .unwrap();
        let score = f32::from_le_bytes(score_bytes);
        tokenizer.vocab_scores.push(score);

        // Read string length
        let mut len_bytes = [0u8; size_of::<i32>()];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "failed read"))
            .unwrap();
        let len = i32::from_le_bytes(len_bytes) as usize;

        // Read string data
        let mut string_bytes = vec![0u8; len];
        reader
            .read_exact(&mut string_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "failed read"))
            .unwrap();

        // Convert bytes to UTF-8 string
        let vocab_string = String::from_utf8(string_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 in vocab"))
            .unwrap();

        tokenizer.vocab.push(vocab_string);
    }
}

fn decode(t: &mut Tokenizer, prev_token: i32, token: i32) -> String {
    let mut piece = t.vocab[token as usize].clone(); // Get a String, not &str

    // Following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if prev_token == 1 && piece.starts_with(' ') {
        piece = piece[1..].to_string(); // Remove the & here
    }

    // Check for hex byte pattern
    if let Some(byte_val) = parse_hex_byte(&piece) {
        // Add & here for the function call
        // Return the single byte as a string
        return String::from_utf8_lossy(&[byte_val]).to_string();
    }

    piece // Already a String, no need for .to_string()
}

fn parse_hex_byte(s: &str) -> Option<u8> {
    // Check if string matches pattern "<0xHH>"
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        // Extract the hex part (characters 3-4)
        let hex_part = &s[3..5];
        u8::from_str_radix(hex_part, 16).ok()
    } else {
        None
    }
}

fn safe_printf(piece: Option<&str>) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.

    // Handle None case (equivalent to NULL in C)
    let piece = match piece {
        Some(p) => p,
        None => return,
    };

    // Handle empty string
    if piece.is_empty() {
        return;
    }

    // If piece is a single character, check if it's printable
    if piece.len() == 1 {
        let byte_val = piece.bytes().next().unwrap();
        if !(byte_val.is_ascii_graphic() || byte_val.is_ascii_whitespace()) {
            return; // bad byte, don't print it
        }
    }

    print!("{}", piece);
    io::stdout().flush().unwrap(); // Ensure immediate output like printf
}

fn str_lookup(str: &str, sorted_vocab: &[TokenIndex], vocab_size: usize) -> i32 {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found

    // Use binary search to find the token
    match sorted_vocab.binary_search_by(|token| token.str.cmp(&str.to_string())) {
        Ok(index) => sorted_vocab[index].id as i32,
        Err(_) => -1,
    }
}

// encode the string text (input) into an upper-bound preallocated tokens[] array
// bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
fn encode(t: &mut Tokenizer, text: &str, bos: bool, eos: bool, tokens: &mut Vec<i32>) -> usize {
    // Clear the tokens vector to start fresh
    tokens.clear();

    // Initialize sorted vocabulary if needed (lazy initialization)
    if t.sorted_vocab.is_none() {
        // Create TokenIndex entries for all vocabulary items
        let mut sorted_vocab = Vec::with_capacity(t.vocab_size as usize);
        for (i, vocab_str) in t.vocab.iter().enumerate() {
            sorted_vocab.push(TokenIndex {
                str: vocab_str.to_string(),
                id: i,
            });
        }
        // Sort vocabulary by string for binary search
        sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));
        t.sorted_vocab = Some(sorted_vocab);
    }

    // Create a temporary buffer for merge candidates
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    let buffer_size = (t.max_token_length * 2 + 1 + 2) as usize;
    let mut str_buffer = String::with_capacity(buffer_size);

    // Add optional BOS (=1) token, if desired
    if bos {
        tokens.push(1);
    }

    // Add dummy prefix token if text is not empty
    // This prepends a space token to match SentencePiece behavior
    if !text.is_empty() {
        if let Some(ref sorted_vocab) = t.sorted_vocab {
            let dummy_prefix = str_lookup_concise(" ", sorted_vocab);
            if dummy_prefix != -1 {
                tokens.push(dummy_prefix);
            }
        }
    }

    // Process the raw UTF-8 byte sequence of the input string
    let text_bytes = text.as_bytes();
    let mut i = 0;
    let mut str_len = 0;
    let mut byte_buffer = [0u8; 4]; // Buffer for current UTF-8 character

    while i < text_bytes.len() {
        let current_byte = text_bytes[i];

        // Reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (byte & 0xC0) keeps the first 2 bits
        // 0x80 is 10000000 - UTF-8 continuation bytes start with "10"
        // So this checks: "if this byte is NOT a continuation byte"
        if (current_byte & 0xC0) != 0x80 {
            // This byte is either a leading byte (11...) or ASCII (0x...)
            // Reset our position as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // Append the current byte to the buffer
        byte_buffer[str_len] = current_byte;
        str_len += 1;

        // Check if next character is a continuation byte
        let next_is_continuation = if i + 1 < text_bytes.len() {
            (text_bytes[i + 1] & 0xC0) == 0x80 && str_len < 4
        } else {
            false
        };

        // If next byte is continuation, keep collecting bytes
        if next_is_continuation {
            i += 1;
            continue;
        }

        // We've collected a complete UTF-8 codepoint
        // Convert bytes to string and look up in vocabulary
        if let Ok(utf8_str) = std::str::from_utf8(&byte_buffer[..str_len]) {
            if let Some(ref sorted_vocab) = t.sorted_vocab {
                let id = str_lookup_concise(utf8_str, sorted_vocab);

                if id != -1 {
                    // Found this codepoint in vocab, add it as a token
                    tokens.push(id);
                } else {
                    // Byte fallback encoding: encode each byte as a token
                    // +3 because first 3 vocab elements are <unk>, <s>, </s>
                    for j in 0..str_len {
                        tokens.push(byte_buffer[j] as i32 + 3);
                    }
                }
            }
        }

        str_len = 0; // Reset for next character
        i += 1;
    }

    // Merge phase: find the best consecutive pair to merge iteratively
    loop {
        let mut best_score = -1e10_f32;
        let mut best_id = -1_i32;
        let mut best_idx = -1_isize;

        // Look for the best pair to merge
        for i in 0..(tokens.len().saturating_sub(1)) {
            // Create concatenated string of two consecutive tokens
            str_buffer.clear();
            str_buffer.push_str(&t.vocab[tokens[i] as usize]);
            str_buffer.push_str(&t.vocab[tokens[i + 1] as usize]);

            // Check if this merged string exists in vocabulary
            if let Some(ref sorted_vocab) = t.sorted_vocab {
                let id = str_lookup_concise(&str_buffer, sorted_vocab);
                if id != -1 {
                    let score = t.vocab_scores[id as usize];
                    if score > best_score {
                        // This merge pair exists and has better score
                        best_score = score;
                        best_id = id;
                        best_idx = i as isize;
                    }
                }
            }
        }

        // If no good merge found, we're done
        if best_idx == -1 {
            break;
        }

        // Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx as usize] = best_id;

        // Remove token at position best_idx+1 by shifting everything left
        tokens.remove(best_idx as usize + 1);
    }

    // Add optional EOS (=2) token, if desired
    if eos {
        tokens.push(2);
    }

    tokens.len()
}

// Helper function for string lookup (already defined earlier)
fn str_lookup_concise(str: &str, sorted_vocab: &[TokenIndex]) -> i32 {
    match sorted_vocab.binary_search_by_key(&str, |token| &token.str) {
        Ok(index) => sorted_vocab[index].id as i32,
        Err(_) => -1,
    }
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

pub struct ProbIndex {
    prob: f32,
    index: usize, // struct used when sorting probabilities during top-p sampling
}

pub struct Sampler {
    pub vocab_size: usize,
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
    pub probindex: Vec<ProbIndex>,
}

fn sample_argmax(prob: &[f32], n: usize) -> i32 {
    let mut max_i: i32 = 0;
    let mut max_p = prob[0];

    for i in 1..n {
        if prob[i] > max_p {
            max_i = i as i32;
            max_p = prob[i];
        }
    }
    max_i
}

fn sample_mult(prob: &[f32], n: usize, coin: f32) -> i32 {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()

    let mut cdf = 0.0f32;
    for i in 0..n {
        cdf += prob[i];
        if coin < cdf {
            return i as i32;
        }
    }

    return (n as i32) - 1;
}

fn compare(a: &ProbIndex, b: &ProbIndex) -> Ordering {
    // reversed comparison for descending order (higher prob first)
    b.prob.partial_cmp(&a.prob).unwrap_or(Ordering::Equal)
}

fn sample_topp(
    probabilities: &[f32],
    n: usize,
    topp: f32,
    probindex: &mut Vec<ProbIndex>,
    coin: f32,
) -> usize {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    let n = probabilities.len();
    probindex.clear(); // Start with empty vector

    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    let cutoff = (1.0f32 - topp) / (n - 1) as f32;

    for (i, &prob) in probabilities.iter().enumerate() {
        if prob >= cutoff {
            probindex.push(ProbIndex { index: i, prob });
        }
    }

    // Sort in descending order of probabilities
    probindex.sort_by(compare);

    let n0 = probindex.len();
    if n0 == 0 {
        return 0; // Fallback if no tokens meet cutoff
    }

    // truncate the list where cumulative probability exceeds topp
    let mut cumulative_prob = 0.0f32;
    let mut last_idx = n0 - 1; // in case of rounding errors consider all elements

    for (i, prob_item) in probindex.iter().enumerate() {
        cumulative_prob += prob_item.prob;
        if cumulative_prob > topp {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    let r = coin * cumulative_prob;
    let mut cdf = 0.0f32;

    for i in 0..=last_idx {
        cdf += probindex[i].prob;
        if r < cdf {
            return probindex[i].index;
        }
    }

    probindex[last_idx].index // in case of rounding errors
}

impl Sampler {
    fn new(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        Sampler {
            vocab_size,
            temperature,
            topp,
            rng_state: rng_seed,
            // Pre-allocate buffer for nucleus sampling
            probindex: Vec::with_capacity(vocab_size),
        }
    }

    fn sample(&mut self, logits: &mut [f32]) -> i32 {
        // sample the token given the logits and some hyperparameters
        let next;

        if self.temperature == 0.0f32 {
            // greedy argmax sampling: take the token with the highest probability
            next = sample_argmax(logits, self.vocab_size);
        } else {
            // apply the temperature to the logits
            for q in 0..self.vocab_size {
                logits[q] /= self.temperature;
            }

            // apply softmax to the logits to get the probabilities for next token
            softmax(logits, self.vocab_size);

            // flip a (float) coin (this is our source of entropy for sampling)
            let coin = random_f32(&mut self.rng_state);

            // we sample from this distribution to get the next token
            if self.topp <= 0.0 || self.topp >= 1.0 {
                // simply sample from the predicted probability distribution
                next = sample_mult(logits, self.vocab_size, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sample_topp(
                    logits,
                    self.vocab_size,
                    self.topp,
                    &mut self.probindex,
                    coin,
                ) as i32;
            }
        }

        next
    }
}

fn build_sampler(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> Sampler {
    Sampler::new(vocab_size, temperature, topp, rng_seed)
}

fn random_f32(state: &mut u64) -> f32 {
    // Update the RNG state
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);

    // Convert to float in [0, 1)
    ((*state >> 16) & 0x7fff) as f32 / 32768.0
}

// ----------------------------------------------------------------------------
// generation loop

fn generate(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    prompt: Option<&str>,
    steps: usize,
) -> io::Result<()> {
    // Handle null prompt
    let prompt = prompt.unwrap_or("");

    // encode the (string) prompt into tokens sequence
    let mut prompt_tokens = Vec::with_capacity(prompt.len() + 3); // +3 for '\0', ?BOS, ?EOS
    let num_prompt_tokens = encode(tokenizer, prompt, true, false, &mut prompt_tokens);

    if num_prompt_tokens < 1 {
        eprintln!("something is wrong, expected at least 1 prompt token");
        std::process::exit(1);
    }

    // start the main loop
    let mut start = 0u128; // used to time our code, only initialized after first iteration
    let mut next: usize; // will store the next token in the sequence
    let mut token = prompt_tokens[0] as usize; // kick off with the first token in the prompt
    let mut pos = 0; // position in the sequence

    while pos < steps {
        // forward the transformer to get logits for the next token
        let logits = forward(transformer, token, pos);

        // advance the state machine
        if pos < num_prompt_tokens - 1 {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1] as usize;
        } else {
            // otherwise sample the next token from the logits
            // Convert logits to mutable slice for sampling
            let mut logits_mut = logits.to_vec();
            next = sampler.sample(&mut logits_mut) as usize;
        }
        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next == 1 {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        let piece = decode(tokenizer, token as i32, next as i32);
        safe_printf(Some(&piece)); // same as printf("%s", piece), but skips "unsafe" bytes
        io::stdout().flush().unwrap();
        token = next;

        // init the timer here because the first iteration can be slower
        if start == 0 {
            start = time_in_ms();
        }
    }
    println!(); // Print newline

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if pos > 1 {
        let end = time_in_ms();
        let elapsed_ms = end - start;
        let tokens_per_second = (pos - 1) as f64 / (elapsed_ms as f64 / 1000.0);
        eprintln!("achieved tok/s: {:.2}", tokens_per_second);
    }

    Ok(())
}

fn read_stdin(guide: &str, bufsize: usize) -> io::Result<String> {
    // read a line from stdin, up to but not including \n
    print!("{}", guide);
    io::stdout().flush()?;

    let stdin = io::stdin();
    let mut buffer = String::with_capacity(bufsize);

    match stdin.lock().read_line(&mut buffer) {
        Ok(_) => {
            // Remove trailing newline if present
            if buffer.ends_with('\n') {
                buffer.pop();
                if buffer.ends_with('\r') {
                    buffer.pop(); // Handle Windows line endings
                }
            }
            Ok(buffer)
        }
        Err(e) => Err(e),
    }
}

fn time_in_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_millis()
}

fn chat(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    cli_user_prompt: Option<&str>,
    cli_system_prompt: Option<&str>,
    steps: usize,
) -> io::Result<()> {
    let mut prompt_tokens = Vec::with_capacity(1152);
    let mut user_idx = 0;
    let mut user_turn = true;
    let mut pos = 0;
    let mut next = 0;

    while pos < steps {
        if user_turn {
            let (system_prompt, user_prompt) =
                get_prompts(pos == 0, cli_system_prompt, cli_user_prompt)?;

            let rendered_prompt = render_chat_prompt(pos == 0, &system_prompt, &user_prompt);

            let num_prompt_tokens =
                encode(tokenizer, &rendered_prompt, true, false, &mut prompt_tokens);
            if num_prompt_tokens == 0 {
                eprintln!("Warning: No tokens generated from prompt");
                continue;
            }

            user_idx = 0;
            user_turn = false;
            print!("Assistant: ");
            io::stdout().flush()?;
        }

        let token = if user_idx < prompt_tokens.len() {
            let token = prompt_tokens[user_idx] as usize;
            user_idx += 1;
            token
        } else {
            next
        };

        // EOS token ends Assistant turn
        if token == 2 {
            user_turn = true;
            continue;
        }

        let logits = forward(transformer, token, pos);
        let mut logits_mut = logits.to_vec();
        next = sampler.sample(&mut logits_mut) as usize;
        pos += 1;

        // Print Assistant response
        if user_idx >= prompt_tokens.len() && next != 2 {
            let piece = decode(tokenizer, token as i32, next as i32);
            safe_printf(Some(&piece));
            io::stdout().flush()?;
        }

        if next == 2 {
            println!();
        }
    }

    println!();
    Ok(())
}

fn get_prompts(
    is_first_turn: bool,
    cli_system_prompt: Option<&str>,
    cli_user_prompt: Option<&str>,
) -> io::Result<(String, String)> {
    let system_prompt = if is_first_turn {
        if let Some(sys_prompt) = cli_system_prompt {
            sys_prompt.to_string()
        } else {
            read_stdin("Enter system prompt (optional): ", 512)?
        }
    } else {
        String::new()
    };

    let user_prompt = if is_first_turn && cli_user_prompt.is_some() {
        cli_user_prompt.unwrap().to_string()
    } else {
        read_stdin("User: ", 512)?
    };

    Ok((system_prompt, user_prompt))
}

fn render_chat_prompt(is_first_turn: bool, system_prompt: &str, user_prompt: &str) -> String {
    if is_first_turn && !system_prompt.is_empty() {
        format!(
            "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
            system_prompt, user_prompt
        )
    } else {
        format!("[INST] {} [/INST]", user_prompt)
    }
}
fn load_tokenizer(path: &str, expected_vocab_size: i32) -> io::Result<Tokenizer> {
    let mut bytes = Vec::new();
    File::open(Path::new(path))?.read_to_end(&mut bytes)?;
    if bytes.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tokenizer file too small: {} bytes", bytes.len()),
        ));
    }

    // Try parse with/without header
    let try_parse = |start_off: usize, count: Option<usize>| -> io::Result<Tokenizer> {
        let mut t = Tokenizer {
            vocab_size: 0,
            vocab: Vec::new(),
            vocab_scores: Vec::new(),
            sorted_vocab: None,
            max_token_length: 0,
            byte_pieces: [0u8; 512],
        };
        for i in 0..256 {
            t.byte_pieces[i * 2] = i as u8;
            t.byte_pieces[i * 2 + 1] = 0;
        }

        let mut off = start_off;
        let mut tokens = 0usize;
        let hard_cap = count.unwrap_or(usize::MAX);

        while off + 8 <= bytes.len() && tokens < hard_cap {
            // score
            let sb: [u8; 4] = bytes[off..off + 4].try_into().unwrap();
            off += 4;
            let score = f32::from_le_bytes(sb);

            // len
            let lb: [u8; 4] = bytes[off..off + 4].try_into().unwrap();
            off += 4;
            let len = i32::from_le_bytes(lb);
            if len < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("negative string length at token {}", tokens),
                ));
            }
            let len = len as usize;

            // sanity: cap individual token length
            if len > 1_000_000 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unreasonable string length {} at token {}", len, tokens),
                ));
            }
            if off + len > bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "EOF while reading string payload of token {} (need {} bytes, have {})",
                        tokens,
                        len,
                        bytes.len() - off
                    ),
                ));
            }

            let s = std::str::from_utf8(&bytes[off..off + len])
                .map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid UTF-8 at token {}", tokens),
                    )
                })?
                .to_string();
            off += len;

            t.vocab_scores.push(score);
            t.vocab.push(s);
            tokens += 1;
        }

        t.vocab_size = t.vocab.len() as i32;
        if let Some(expected) = count.map(|c| c as i32) {
            if t.vocab_size != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "parsed {} tokens but expected {} (layout start_off={})",
                        t.vocab_size, expected, start_off
                    ),
                ));
            }
        }
        t.max_token_length = t.vocab.iter().map(|s| s.len() as u32).max().unwrap_or(0);
        Ok(t)
    };

    // Read possible header vocab_size
    let header_vocab = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let has_reasonable_header = header_vocab > 0 && header_vocab < 1_000_000;

    // Strategy:
    // 1) If header looks reasonable, try Layout B using header_vocab.
    // 2) If that fails, try Layout A with expected_vocab_size.
    // 3) If that fails, try Layout A w/out forcing count (consume all).
    if has_reasonable_header {
        if let Ok(tok) = try_parse(size_of::<i32>(), Some(header_vocab as usize)) {
            // Optional: verify it matches model header (warn if not equal)
            if header_vocab != expected_vocab_size {
                eprintln!(
                    "⚠️ tokenizer header vocab_size ({}) != model header vocab_size ({}). Using tokenizer value.",
                    header_vocab, expected_vocab_size
                );
            }
            return Ok(tok);
        }
    }

    if let Ok(tok) = try_parse(0, Some(expected_vocab_size as usize)) {
        return Ok(tok);
    }

    // Fallback: parse entire file without a forced count (helps debugging)
    let tok = try_parse(0, None)?;
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "could not parse tokenizer with expected vocab {}, parsed {} tokens instead",
            expected_vocab_size, tok.vocab_size
        ),
    ))
}

fn load_transformer(checkpoint_path: &str) -> io::Result<Transformer<'static>> {
    let path = Path::new(checkpoint_path);

    // read header + mmap file
    let (cfg, mmap, shared_weights) = read_checkpoint(path)?;

    // ---- weights slice out of the mmap payload ----
    let weights_bytes = &mmap[size_of::<Config>()..];

    if weights_bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "weights length not multiple of 4",
        ));
    }

    // SAFETY: checkpoint format is little-endian f32-packed after header.
    // If you want to be extra safe, also check alignment via align_to.
    let weights_f32: &'static [f32] = unsafe {
        std::slice::from_raw_parts(
            weights_bytes.as_ptr() as *const f32,
            weights_bytes.len() / 4,
        )
    };

    // ---- prepare an empty weights container and map in-place ----
    let mut weights = TransformerWeights {
        token_embedding_table: &[],
        rms_att_weight: &[],
        rms_ffn_weight: &[],
        wq: &[],
        wk: &[],
        wv: &[],
        wo: &[],
        w1: &[],
        w2: &[],
        w3: &[],
        rms_final_weight: &[],
        wcls: &[],
    };

    memory_map_weights(&mut weights, &cfg, weights_f32, shared_weights as i32);

    // ---- allocate runtime state buffers (owned -> leak to get 'static refs) ----
    let dim = cfg.dim as usize;
    let hidden_dim = cfg.hidden_dim as usize;
    let seq_len = cfg.seq_len as usize;
    let n_layers = cfg.n_layers as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let kv_dim = ((cfg.dim * cfg.n_kv_heads) / cfg.n_heads) as usize;

    // helper to leak a Vec and get &'static mut [f32]
    fn leak_mut_slice(mut v: Vec<f32>) -> &'static mut [f32] {
        let b = v.into_boxed_slice();
        Box::leak(b)
    }

    let x = leak_mut_slice(vec![0.0; dim]);
    let xb = leak_mut_slice(vec![0.0; dim]);
    let xb2 = leak_mut_slice(vec![0.0; dim]);
    let hb = leak_mut_slice(vec![0.0; hidden_dim]);
    let hb2 = leak_mut_slice(vec![0.0; hidden_dim]);
    let q = leak_mut_slice(vec![0.0; dim]);
    let k = leak_mut_slice(vec![0.0; dim]);
    // NOTE: your RunState defines `v: &'a [f32]` (immutable),
    // but you write to value cache. Keep it `&'a mut [f32]` if you plan to mutate.
    // For now, provide a dummy immutable slice; your code uses `value_cache` for writes.
    let v_slice: &'static [f32] = {
        let tmp = vec![0.0f32; dim];
        let bx = tmp.into_boxed_slice();
        Box::leak(bx)
    };
    let att = leak_mut_slice(vec![0.0; n_heads * seq_len]);
    let logits = leak_mut_slice(vec![0.0; cfg.vocab_size as usize]);

    // KV cache: (layer, seq_len, kv_dim)
    let key_cache = leak_mut_slice(vec![0.0; n_layers * seq_len * kv_dim]);
    let value_cache = leak_mut_slice(vec![0.0; n_layers * seq_len * kv_dim]);

    let state = RunState {
        x,
        xb,
        xb2,
        hb,
        hb2,
        q,
        k,
        v: v_slice,
        att,
        logits,
        key_cache,
        value_cache,
    };

    // build final Transformer
    Ok(Transformer {
        config: cfg,
        weights,
        _mmap: mmap, // keep mapping alive as long as Transformer lives
        state,
    })
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or("model.bin");
    let tokenizer_path = args.get(2).map(String::as_str).unwrap_or("tokenizer.bin");
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let temperature: f32 = 0.8;
    let topp: f32 = 0.9;
    let seed: u64 = 12345;

    let mut transformer = load_transformer(model_path)?;

    let mut tokenizer = load_tokenizer(tokenizer_path, transformer.config.vocab_size)?;

    let mut sampler = Sampler::new(
        transformer.config.vocab_size as usize,
        temperature,
        topp,
        seed,
    );

    chat(
        &mut transformer,
        &mut tokenizer,
        &mut sampler,
        None,
        None,
        steps,
    )
}
