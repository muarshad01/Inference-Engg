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

||
|---|
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)|

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

* $LM \rightarrow T1 \rightarrow \underset{MLP \rightarrow DT1} \rightarrow MLP \rightarrow DT2 \rightarrow MLP \rightarrow DT3 \rightarrow$
