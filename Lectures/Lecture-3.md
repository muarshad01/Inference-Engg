## Lecture-3
* MHA
* MQA
* GQA
* MLA
* [DeepSeek V3.2 Sparse Attention (DSA)](DeepSeek Sparse Attention From Scratch)

|| Paper | 
|---|---|
| Dec 2025 | [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) |

***

#### MHA
* [Lecture09 - MHA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture09_notes.md)

* 30:00

#### MQA
* [Lecture10 - MQA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture10_notes.md)

***

* 50:00

#### GQA
* [Lecture11 - GQA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture11_notes.md)

***

#### MLA

* [MLA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture12_notes.md)

* Shift focus from reducing the number-of-heads to compressing the informtion within these heads.
* What if we don't have to cache K & V seperately.
* What if, we could first project our input (X) into a single, combined, much smaller matrix, a latent matrix $C_{KV}$ and cache only that!
* This is the central idea of MLA:
* Instead of caching two large matrices, K & V, we only cache one smaller, lower dimensional matrix $C_{KV}$.
* This single matrix becomes our highly efficient cache.
* When we need the full Keys ($K$) and Values($V$), we can resonstruct them on-the-fly from the compressed latent representation ($C_{KV}$).

***

#### Weights versus Activations
* Weights are parameters, which are trained
* In backpropogation, we have to take partial derivates both wrt to Weights and Activations.
* FLOPs (backward psss) = 2 x FLOPS (forward pass)


|||
|---|---|
| scores = $\frac{Q \times K^T}{\sqrt{d}}$ | produces $N \times N$ score matrix |
| weights = softmax(scores)                | also $N \times N$       |
| context = $\text{weights} \times V$                    | producing $N \times d$  |


* 2:10:00

#### Coding
* [roneneldan/TinyStories · Datasets at Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories)

***

* 2:30:00

#### Generate Code through Codex-5.5
* Become a good Agentic Orchestrator to produce  best quality code!

* [Latent Multi-Head Attention for Small Language Models](https://arxiv.org/pdf/2506.09342)

<img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-3/codex-1.png" width="300" height="300" />

<img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-3/codex-2.png" width="300" height="300" />

* You don't have to write a single line of code!

***

* 2:40:00

#### DeepSeek Sparse Attention (DSA)
* Tokens + Index-Key
* MLC Cache ($C_{KV}$) + Indexing Cache($W_Q^I$ and $W_K^I$)
* Its derived from Latent Attention

***

#### Now the new token "bright" comes in

#### MLA Query
* $Q_{bright}   = x_{bright} \times W_Q$
#### MLA Latent Vector
* $cKV_{bright} = x_{bright} \times W_{dKV}$
#### Index Query and Key
* $QI_{bright} = h_{bright} \times W_Q^I$ 
* $KI_{bright} = h_{bright} \times W_K^I$

***

* $KI_{bright}.KI_{s}=\text{score}(bright, s)$

***

* 2:50:00

* DeepSeek seq-length=128K
* New query -> 128K latent vectors need to be loaded from cache.
* All the past is not important
* I want to load top 2,048 latent vectors only.
* Instead of reading all 128k past vectors for a new query, you only read 2,048.
* How would you select those 2048?

* 
* Indexer:
* For ALL past tokens, you maintain a low-dimensional vector called the key-indexer.
* For a new Query, you take dot product with ALL 128k key-indexers.
* Read top 2,048 from HRAM.

***

| Method | Cache Formula | Meaning | 
|---|---|---|
| MHA |$l \times b \times s \times \underbrace{n_{heads} \times h}_{\text{embedding dim}} \times 2 \times bytes$ | Sotre K and V for every head. |
| MQA | $l \times b \times s \times \underbrace{1\times h}_{\text{embedding dim}} \times 2 \times bytes$ | All query head share one KV head. |
| GQA | $l \times b \times s \times \underbrace{g \times h}_{\text{embedding dim}} \times 2 \times bytes$ | Groups of query heads share KV heads. |
| MLA | $l \times b \times s \times d_{latent} \times bytes $ | Cache compressed latent content insted of |
| DSA over MLA | $\text{MLA Cache} + l \times b \times s \times d_{index} \times \text{index bytes}$ | Add a lookup cache. The expensive MLA read uses only top-k selectors |
| Sliding Window | $l \times b \times W \times \text{cache per token}$|

***

* SLA is worst the DSA

***

#### World model inference

***

* 3:20:00

*** 
