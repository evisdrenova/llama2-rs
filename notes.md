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

### read_checkpoint

This is a helper function to load an external model so we can use it. We effectively, get a file, read it, and then pass it into our memory map function in order to initialize our model.

//TODO: COMPLETE

## Neural net blocks

Then we start converting some nice pure functions to rust. RMSNorm, Softmax and the star of the show, matmil. These are the critical components to making our neural net work.
