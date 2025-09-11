# llama2-rs

This is a rust port of Karpathy's llama2.c tiny and simple inference engine written in pure c. I thought it would be fun to port this to rust to improve my own learning. I also wanted to take some notes of my thought process as im doing this.

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

First, we'll define our structs here that will hold our token indexes as as well our tokenizer. Then we'll get to work on the helper functions.
