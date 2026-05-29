## Speculative Decoding
* Inference:
  * Prefill (compute bound)
  * decode (memory bound)

***

* 15:00

#### Example: Memory Bound

|||
|---|---|
| Tiny Llama    | 1.1 B |
| Model Weights | 2.2 GB |
| HBM (A100)    | 2 TB/s |
|Time to download these wts to compute area | $\frac{2.2 ~GB}{2 ~TB/s} = 1.1 ~ms$ |
| FLOPs for 1 token decoding | 2 x num_params = 2.2 B|
| New Token: $x(1, m)$      |  |
| Weight query matrix: $W_q(m, n)$   ||
| FLOPs for $x(1, m) \times W_q(m, n)$   | 2 x mn |
| num_params $(W_q) = (m,n)$             | 2 x num_params |
| A200 | 312 TFLOPs |
| Time to decode one token | $\frac{2.2 B}{312 TFLOPs}=0.007~ms$|
| Ratio = 1.1ms (loading) / 0.007 ms (decoding) = 157 | GPU spends 99% time loading and only 1% time computing! |

***

* 30:00

* LLM decoding: autoregressive!

* Someone (draft model) guesses (speculating; fater and parallel) the future tokens for us!

## ##Draft Models
1. N-gram model - look at a similar repetitive patter in the sequence
2. EAGLE: Extrapolation Algorithm for Greater Language-model Efficiency
3. Medusa

| Paper |
|---|
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)|
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads - Jun 2024](https://arxiv.org/abs/2401.10774)|
* Idea: Language model repeat patterns.

#### Tiny Llama (~1.1B)
* Embedding layer (v_size, embed_dim) = (32,000, 2,048)
* Transformer layers L = 22
* LM head (2,048, 32,000)

#### Draf model
* Fast, small, related to main model somehow.
* EAGLE has also similar embedding layer
* Embedding layer (v_size, embed_dim) = (32,000, 2,048) - same as Large model
* Lightweight NN: (2-layer; hidden dim=512) - 1% or original parameters
* LM head (2,048, 32,000) - same as Large model

*** 

* **Step-1**: Remove the 22 transformer layers from TinyLlama. Replace them with small MLP (2 linear layers with ReLU). Keep the embedding and LM head from the original model.
* **Step-2**: Train the small MLP to mimic the full model behavior.
* **Training data**: Run the large model, record the input embeddings and the output logits.
* The MLP learns to approximate the logits from the embeddings!!

*** 

* Token - T
* Draft Token - DT
* $LM \rightarrow T1 \rightarrow \underbrace{MLP \rightarrow DT1 \rightarrow MLP \rightarrow DT2 \rightarrow MLP \rightarrow DT3}_{3 ~draft ~tokens!}$

#### Why does EAGLE work
* The embeddings capture rich semantic information
* The MLP learns a compressed shortcut from the embedding space to next token space
* Its definately less accurate than the full transformer model, but much faster!

*** 

* 1:05

#### Medusa (Greek Demon)
* EAGLE generated tokens in sequential manner
* What about if tokens are generated in Parallel manner

* We grow extra heads on top of the large model.
* Each head predicts a different future token prediction -> all in parallel, from the same hidden states.

* Medusa embedding layer
* 22 transformer layers
* LM head (original) - t+1
* Medusa head 1 - t+1
* Medusa head 2 - t+2
* Medusa head 3 - t+3

#### How it works
1. **Training**: Freeze large model entirely.
2. Train k extra heads t+2, t+3, t+k+1
3. Training data: Run large model and record future tokens
4. Each head learns: given hidden state at position t, what is token t+i
* **At Runtin**: After each large model forward pass: Original LM head predicts t+1 (as usual)
* Medusa Head 1 predicts - t+2
* Medusa Head 2 predicts - t+3
* Medusa Head 3 predicts - t+4
* All heads run in "parallel" -> they use the same hidden state!

Cost: K extra matrix multiplicatins. Much cheaper compared to 22 layer forward pass.

***

* Medusa predicts draft tokens in parallell at run time.
* EAGLE predicts draft tokens sequentially at run time.

*** 
