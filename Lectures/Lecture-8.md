## Speculative Decoding
* Inference:
  * Prefill (Compute bound)
  * Decode (Memory bound)

***

* 15:00

#### Example: Memory Bound

|||
|---|---|
| Tiny Llama    | 1.1 B |
| Model Weights | 2.2 GB |
| HBM (A100)    | 2 TB/s |
|Time to download these wts to compute area | $\frac{2.2 ~GB}{2 ~TB/s} = 1.1 ~ms$ |
| **FLOPs for 1 token decoding** | **2 x num_params = 2.2 B** |
| New Token: $x(1, m)$      |  |
| Weight query matrix: $W_q(m, n)$   ||
| FLOPs for $x(1, m) \times W_q(m, n) = output(1,n)$   | $2m \times n$ |
| num_params $(W_q) = (m,n)$             | 2 x num_params |
| A100 | 312 TFLOPs |
| Time to decode one token | $\frac{2.2 B}{312 TFLOPs}=0.007~ms$|
| Ratio = 1.1ms (loading) / 0.007 ms (decoding) = 157 | GPU spends 99% time loading and only 1% time computing! |

#### Main Insight:
* If we load weights anyway, why not compute multiple tokens in the same pass?
* The loading cost is the same, but we get more tokens per pass!

***

#### LLM decoding
* Issue is LLM decoing is **Autoregressive**
* We don't know the next token untill we sample it
* We can't comptue tokens 2,3,4 untill we know token 1!!
* If I can't predict the next token, I can at least "guess" it
* I have 1 token
* I guess tokesn 2,3,4
* Then someone will verify my guess in parallel! 

#### Speculative Decoding
* Someone (draft model) guesses (speculating; faster and parallel) the future tokens for us!
* A fancy word for guessing is **speculating**.
* Speculate the future, like a prophecy!
* (Draft, Target) model

***

* 30:00

## Draft Models

#### **1 - Ngram**:
* Simplest draft model.
* It requires no seperate model.
* Looks at a similar repetitive pattern in text already generated.
* **Idea**: Language models often reapeat patterns.
* **Note**: vLLM has Ngram matching.

***

* 45:00

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-8/EAGLE.png" width="500" height="400" />
</p>

#### **2 - EAGLE**:
* Extrapolation Algorithm for Greater Language-model Efficiency (EAGLE)
* Takes the large model and creates a stripped-down version
* We need to predict draft-token but in a much more sensible manner
* Generates draft-tokens in sequential manner

#### Tiny Llama (~1.1B)
* Embedding layer: $(vocab_{size}, embed_{dim}) = (32,000, 2,048)$
* Transformer layers L = 22
* LM head: $(2,048, 32,000)$

#### Draf model
* Fast, small, related to main model somehow.
* EAGLE has also similar embedding layer
* We do a model surgery on large model
* Embedding layer $(vocab_{size}, embed_{dim}) = (32,000, 2,048)$ - same as Large model!
* **Lightweight NN**:
  * 2-layer; hidden_dim=512 - 1% or original parameters
  * We replace 22 Tranfer layers with just 2 layer NN
* LM head $(2,048, 32,000)$ - same as Large model (shared)!

*** 

* **Step-1 (Surgery)**:
  * Remove the 22 transformer layers from TinyLlama. Replace them with small MLP (2 linear layers with ReLU).
  * Keep the embedding and LM head from the original model.
* **Step-2** (Finetuning):
  * Train the small MLP to mimic the full model behavior.
* **Training data**:
  * Run the large model, record the input embeddings and the output logits.
  * The MLP learns to approximate the logits from the embeddings!!

*** 

* **Step-3** (Runtime):
* Here is how we generate draft tokens.
* (a) Take the last tokens embeddings from the large model (already computed).
* (b) Pass it to Tiny MLP -> Get approximate hidden state.
* (c) Pass hidden state through shared LM head -> get draft logits.
* (d) Sampel the first draft token.
* (e) Repeat k times to get k draft tokens.
* Token - T
* Draft Token - DT
* $LM \rightarrow T1 \rightarrow \underbrace{MLP \rightarrow DT1 \rightarrow MLP \rightarrow DT2 \rightarrow MLP \rightarrow DT3}_{3 ~draft ~tokens!}$

#### Why does EAGLE work?
* The embeddings capture rich semantic information
* The MLP learns a compressed shortcut from the **embedding space to next-token space**
* Its definately less accurate than the full transformer model, but much faster!

#### Comparision
* Large model forward pass: 2.2 Billion FLOPS
* EAGLE MLP forward: 20 million FLOPS
* Even if you're decoing 5 tokens, 100 million FLOPS
* Still 22 time cheaper than one large step through the big model.

*** 

* 1:00:00

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-8/medusa.png" width="500" height="400" />
</p>

#### **3 - Medusa**: 
* Generates draft-tokens in **parallel manner**

#### Medusa (Greek Demon)
* EAGLE generated tokens in **sequential manner**
* What about if tokens are generated in Parallel manner

```
* We grow extra heads on top of the large model.
* Each head predicts a different future token position -> all in parallel, from the same hidden states.
```

#### Medusa Architecture
* Medusa embedding layer (Same as orig model)
* 22 transformer layers  (Same as orig model)
* LM head       - t+1    (Same as orig model)  
* Medusa head 1 - t+2
* Medusa head 2 - t+3
* Medusa head 3 - t+4

#### How it works
* **Training**: Freeze large model entirely.
  * Train k extra heads to predict tokens t+2, t+3, t+k+1
  * Training data: Run large model and record future tokens
  * Each head learns: given hidden state at position t, what is token t+i
* **At Runtime**: After each large model forward pass:
  * Original LM head predicts - t+1 (as usual)
  * Medusa Head 1 predicts    - t+2
  * Medusa Head 2 predicts    - t+3
  * Medusa Head 3 predicts    - t+4
* All heads run in "parallel" -> they use the same hidden state!

* **Cost**: K extra matrix multiplicatins.
  * Much cheaper compared to 22 layer forward pass.

***

#### Medusa versus EAGLE
* Medusa predicts draft tokens in parallell at run time.
* EAGLE predicts draft tokens sequentially at run time.

*** 

* 1:25:00

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-8/deepseek-1.png" width="500" height="400" />
</p>

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-8/deepseek-2.png" width="500" height="400" />
</p>

#### DeepSeek
* Implemented MTP during pre-training itself.
* MTP objective densifies the training signals and improves data efficiency.

*** 
* 1:35:00

#### Accept/Reject Algorithm
* Large model
* Draft model

***

* p_draft(x): probability draft model assigns to token x
* p_large(x): probability large model assigns to token x

***

* What if we do a forward pass of these tokens on our target model?
```
If (p_large > p_draft):
   always accept!
```

|   |      t+2  | t+3 | t+4 |
|---|---|---|---|
| p_draft | 0.3 | 0.5 | 0.8 |
| p_large | 0.4 | 0.6 | 0.2 |

* Large model likes this token at least as much as the draft!

*** 

#### Probabilistic Sampling
```
Draw random number $r ~ U(0,1)$
If r < p_large/p_draft:
   accpet
If r > p_large/p_draft:
   reject
```

* Rebalance distribution
* Draft bias

*** 

* 2:05:00

* The Accept/Reject Rule Preserves the Target Distribution

|||
|---|---|
| If $p_{target} \ge p_{draft}$ | ACCEPT with probability 1. |
| If $p_{target} < p_{draft}$    | ACCEPT with probability $\frac{p_{target}}{p_{draft}}$. |
| If REJECTED                    | Resample from the adjusted distribution $\text{max}(0, p_{target}-p_{draft})$, normalized |

***

* [Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
* DFlash

*** 

* 2:20:00


| Paper |
|---|
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Jan 2024)](https://arxiv.org/abs/2401.15077) |
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads - Jun 2024](https://arxiv.org/abs/2401.10774) |
| [Fast Inference from Transformers via Speculative Decoding - May 2023](https://arxiv.org/abs/2211.17192) |
| [Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation - Dec 2024](https://arxiv.org/abs/2406.02347) |

***

#### [Speculative Decoding: When Two LLMs are Faster than One](https://www.youtube.com/watch?v=S-8yr_RibJ4)

***
