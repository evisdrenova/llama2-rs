# llama2-rs

This is a rust port of [Karpathy's llama2.c tiny and simple inference engine written in pure c](https://github.com/karpathy/llama2.c). I thought it would be fun to port this to rust to improve my own learning. I also wanted to take some notes of my thought process as I'm doing this.

I'm following this file top to bottom so my notes will flow in that direction.

here we go!

## Converting the structs

First thing first, let's convert the structs. Most of this is pretty straightforward but some things to note. In C, this:

`float *token_embedding_table; // (vocab_size, dim)`

Can be interpreted by the compiler as a single float value or an element in a contiguous array of floats. So in rust, we translate this to:

`pub token_embedding_table: Vec<f32>,`

This gives us a growable contiguous array of float32 values. The `float` keyword in C is almost always a f32. Alternatively, we could define it as:

`pub token_embedding_table: Vec<c_float>,`

if we wanted it to be interoperable with C but since this is just a rust project, we'll keep it as a native rust type.

`int` in C is a signed int (usually 32) but since these are just hyper paramters we'll just define them as `usize` to make the downstream work a little easier.

## Core RunState functions

Next, we'll work on some of the core runstate functions i.e. memory_map_weights, read_checkpoint and build_transformer. Since we're doing this in rust, we can skip the calloc and free functions (malloc_run_state, free_run_state, free_transformer). The benefit of using rust is that if we write our code well, we don't have to worry about manually freeing up the memory.

Lets go through each function.

### memory_map_weights

This function slices (no pun intended) the flat weights array into each field that we defined in our `TransformerWeights` struct. We use a pointer to hope around this contiguous array and extract the data that we need and assign it to the right field in our struct so we can do something with it later.

here we're memory mapping the entire model as a flat array of float32s
we use a pointer to jump around our array in memory, it's laid out like this:

```[ token_embedding ][ rms_att ][ wq ][ wk ][ wv ][ wo ]
[ rms_ffn ][ w1 ][ w2 ][ w3 ][ rms_final ][ <skip> ][ wcls ]
```

so here, we're taking our model and jumping to each part adn then assign it to the right fields in the struct so we can do stuff with it later.

the C code is: ptr += p->vocab_size \* p->dim;
this moves the pointer forward in memory to get the next contiguous block. we have two options here:

1. use a raw pointer which is unsafe but would exactly mirror the C code
2. create a slice to the contiguous block which chops the block in half at the location calculated by p.vocab_size \* p.dim and return the pointer to the second piece

I went back and forth on whichone to do here and eventually went with the second approach just to be more rust idiomatic but likely i would have preferred the first way.

In order to do this, we define little inline helper function called `take` which takes a slice of floats starting at offset and advance the offset so that the next call starts after it. this is functionally equivalent to using a raw pointer to hop around the weights array but type-safe.

The nice thing is that the buffer in the take is immutable so we don't have to worry abouta ny borrow checker issues since we're just reading the slice and not mutating it in any way.

we should add an assrt here to make sure the ptr is at least as large as the sum of all the sizes calculated inside the function

### read_checkpoint && build_transformer

This is a helper function to load an external model so we can use it. We effectively, get a file, read it, and then pass it into our memory map function in order to initialize our model.

First, we open the file and map it into memory using mmamp. Then we validate that is has a minimum size before we read the config from the beginning of the memory mapped file using byte offsets to parse the different parts of the config.

Lastly, we do some more validation and then return the config, memory mapped file and the shared_weights bool.

In the build_transformer function, we run the read_checkpoint function to read in the file and then we get the weights data from the file and then memory map that to our `TransformerWeights` struct.

Now we have parsed the important bits from the file and loaded it into our structs so we can do stuff with them.

## Neural net blocks

Then we start converting some nice pure functions to rust. RMSNorm, Softmax and the star of the show, matmul. These are straight forward implementations but are key components to the puzzle.

## Forward

This is the forward pass function that we use to pass through the neural net. Let's break this down section by section.

We're running inference for one token at a time at a specific position in the sequence. You can see this since we take in a `token` parameter.

First, we extract some references to the config, weights and run state.

```rust
    let p = &transformer.config;
    let w = &transformer.weights;
    let s = &mut transformer.state;
```

Then, we compute the key dimensions that we will need in the forward pass.

```rust
    let x = s.x;
    let dim = p.dim;
    let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    let kv_mul = p.n_heads / p.n_kv_heads;
    let hidden_dims = p.hidden_dim;
    let head_size = dim / p.n_heads;
```

Then we convert the input token ID to an embedding. The embedding table is `[vocab_size, dim]`, so we index it with `token*dim` to get the embedding for the specific token. The embedding table will be created later as part of the BPE section using the vocab.

```rust
    let start = token * dim;
    let end = start + dim;
    let content_row = &w.token_embedding_table[start..end];
    s.x[..dim].copy_from_slice(&content_row[..dim]);
```

There is a main loop which loops over each transformer layer in in `p.n_layers`.

```rust
    for l in 0..p.n_layers as usize {
    //...
    }
```

Inside our loop, we first normalize the layers using root mean square normalization (RMSNorm), this helps to stabilize training.

```rust
    // attention rmsnorm
    let rms_offset = l * dim;
    let rms_weights = &w.rms_att_weight[rms_offset..rms_offset + dim];
    rmsnorm(&mut s.xb, &s.x, rms_weights, p.dim as usize);
```

Next, we compute the Query, Key and Value vectors by multiplying the normalized input with the learned weight matrices.

```rust
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
```

This is a critical part of our attention calculation and is what helps our LLM attend to each token in the sequence. It's a little more verbose than the C implementation but that's mainly for readability.

Next, we move onto our RoPE (rotary position encoding).

```rust
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

```

Here we apply RoPE to inject position information into the Q and K vectors. This allows the model to understand relative positions of tokens.

Okay, we're moving on! Now to the main attention bit. This is the core multi-headed attention mechanism.

```rust
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
```

For each attention head in `p.n_heads`:

1. split Q,K,V into smaller heads for parallel processing.
2. compute the attention scores - dot product between query and all previous keys
3. scale the scores by dividing them by `sqrt(head_size)`
4. apply the software to convert the scores to probabilities
5. combine values using attention weights to get outputs

By the end of this process, we will have attention scores that we can project back to the model dimensions and add a residual connection.

Almost done, hang in there!

Next step is the Feed Forward Network (FFN) with SwiGLU. The FFN normalizes the attention scores and then projects it to two linear projections of size `hidden_dim`.

```rust
  rmsnorm(&mut s.xb, &s.x, rms_ffn_weights, p.dim);

    // Two parallel linear projections
    matmul(&mut s.hb, &s.xb, w1_slice, dim, hidden_dim);   // Gate projection
    matmul(&mut s.hb2, &s.xb, w3_slice, dim, hidden_dim);  // Up projection
```

Then we apply switch activation to one project and then multiply it by the gating mechanism.

```rust
   // SwiGLU activation
    for i in 0..hidden_dim {
        let mut val = s.hb[i];
        val *= 1.0f32 / (1.0f32 + (-val).exp());  // SiLU activation
        val *= s.hb2[i];                           // Gated by second projection
        s.hb[i] = val;
    }

    // Down projection
    matmul(&mut s.xb, &s.hb, w2_slice, hidden_dim, dim);
```

Lastly, we down project the scores back to the model dimensions from the hidden dimensions and then add a skip connection.

```rust
   // Down projection
    matmul(&mut s.xb, &s.hb, w2_slice, hidden_dim, dim);

    // Residual connection
    for i in 0..dim {
        s.x[i] += s.xb[i];
    }
```

LAST STEP! Just a simple root mean square normalization and then we project the model to the vocab size to get the logits for the next token prediction and return the logits.

````rust
  rmsnorm(&mut s.x, &s.x.clone(), &w.rms_final_weight, p.dim);

    // Classification head
    matmul(&mut s.logits, &s.x, &w.wcls, p.dim as usize, p.vocab_size as usize);

    &s.logits
    ```
````

## BPE Tokenizer

Next, we'll move onto the byte pair encoder tokenizer. The tokenizer is responsible for taking our input prompt, and chopping it up into tokens derived from our vocab, creating embeddings from those tokens and feeding those embeddings to our Transformer that we created above.

We'll first define our structs that will hold our token indexes as as well our tokenizer and then build the tokenizer. Here is the code:

```rust
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
```

So whats happenign here? This code is creating null-terminated single-byte strings in the byte_pieces array:

```
Index:  0  1  2  3  4  5  6  7  8  9 ...
Value: [0][0][1][0][2][0][3][0][4][0]...
        ^^^^^ ^^^^^ ^^^^^ ^^^^^ ^^^^^
        byte  byte  byte  byte  byte
         0     1     2     3     4
```

Each byte value (0-255) gets stored as a 2-byte null-terminated string. This is the foundation of tokenizer so that we can tokenize any prompt that the user gives.

Next, we convert the decode function, which converts token IDs back into their string representations and handles a couple of cases.

```rust

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
```

It does a basic token lookup in the vocab and it handles whitespace after the BOS (beginning of sequence) token. There are a couple of more helper functions that we convert: safe_printf and str_lookup, then we get to the other big function, encode.

The encode function takes in a prompt and encodes it into a token array that we can then embed.Here's what it does:

1. Splits text into UTF-8 characters
2. Looks up each character in vocabulary
3. Falls back to byte encoding for unknown characters
4. Merges tokens using BPE to find optimal subword segmentation
5. Adds special tokens (BOS/EOS) as needed

Here's the entire function:

```rust
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
```

Let's walk through this step by step.

```rust
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
```

We start by lazily creating the sorted vocab when first needed. This involves iterating over the vocab and for each entry creating a `TokenIndex` with the string and id. Then we sort by the string to enable binary search lookups.

Next, we create a temporary buffer for merge candidates.

```rust
  let buffer_size = (t.max_token_length * 2 + 1 + 2) as usize;
    let mut str_buffer = String::with_capacity(buffer_size);

```

And then add special tokens:

```rust
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

```

Then comes the hard part. This took a while and a lot of help from claude code.

```rust
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
```

This checks if we're at the start of a new character:

0xC0 = 11000000 (binary mask for first 2 bits)
0x80 = 10000000 (continuation bytes start with 10)
If NOT a continuation byte, reset buffer for new character:

```rust
byte_buffer[str_len] = current_byte; str_len += 1;
```

Then we collect bytes that form one UTF-8 character:

```rust
let next_is_continuation = (text_bytes[i + 1] & 0xC0) == 0x80 && str_len < 4;
```

Then we check if next byte is part of same UTF-8 character (max 4 bytes per char)

```rust
let id = str_lookup_concise(utf8_str, sorted_vocab);
if id != -1 { tokens.push(id); }
```

if character not in vocab, encode each byte as token ID (byte value + 3, since first 3 IDs are reserved for special tokens):

```rust
else {
    for j in 0..str_len {
        tokens.push(byte_buffer[j] as i32 + 3); // Byte fallback
    }
}
```

Then we get to the merge phase of the BPE algorithm. Here we iteratively merge the most frequent/highest-scoring consecutive token pairs until no more beneficial merges exist

```rust
loop {
    // Find best consecutive pair to merge
    for i in 0..(tokens.len().saturating_sub(1)) {
        str_buffer = concat(vocab[tokens[i]], vocab[tokens[i+1]]);
        let score = vocab_scores[merged_token_id];
        // Track best scoring merge
    }
}
```

Then we execute the best merge by replacing two tokens with one:

```rust
tokens[best_idx] = best_id; // Replace first token with merged result
tokens.remove(best_idx + 1); // Remove second token
```

Lastly the final step adds the end of seuence token:

```rust
if eos { tokens.push(2); } // End of sequence
```

## Sampler

Nice - we're geting pretty close to being done. Next, we have the sampler. The sampler takes in logits and returns a sampled token from a distribution. There are a few ways to do this: greedy argmax, sampling, top-p sampling and others but we'll focus on these for now.

I'm not going to go in depth on these next few functions,. they're pretty straight forward and just a straight port from C.

Next, we'll create a quick implementation of the Sampler struct and define a few methods.

```rust
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
```

Here we define a `new` method to instantiate a new sampler and then we define the sample method. The sample method chooses the right sample method depending on the params that the user has set.

## Generate and Chat

We finally made it to the last section! We're now ready to implement our generate function which runs the generation function and the chat function which accepts our prompt.

```rust
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
```

There's a lot of code there but it's pretty straightforward. This wraps up our port of llama2.c to rust!

Now let's try and run this thing.

First, we need to download the tokenizer models and binaries. I'm just going to take them from the llama2.c repo and drop them in.

We can try and run our model using:

```bash
cargo run --release -- stories15M.bin tokenizer.bin 500
```

I find that the best way to learn something is to re-implement it in a different language. Especially going from C where everything is so granular, you're really forced to get into the weeds.
