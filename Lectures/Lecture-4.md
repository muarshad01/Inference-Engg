## Lecture-4

#### Four Paradigms for Compressing Attention Across Tokens
1. Full Attention (baseline)
2. Sliding Window Attention (SWA)
3. Linear Attention
4. State Space Models (SSM) - Mamba Architecture

***

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-4/4-paradigms-attentions.png" width="500" height="400" />
</p>

***

#### 2 Ways to Compress KV Cache
* Compress KV-cache across number of heads ($n_{heads}$) - [DeepSeek Sparse Attention (DSA)](https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/)
* Compress KV-cache across tokens

***

#### Why we reduce KV-Cache?
* Bytes transferred from High Bandwidth Memory (HBM) to where the compute happens increases.
* AI-decreases (FLOPS/byptes)
* We slide down the roof-life plot
* We want to move from memory-bound to compute-bound

***

* First approach: Reduce across head dimension
* [MHA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture07_notes.md)
* [MQA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture10_notes.md) - $\frac{1}{n}$ reduction
* [GQA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture11_notes.md) - $\frac{g}{n}$ reduction

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-4/tradeoff.png" width="400" height="300" />
</p>

***

#### How to find number of groups

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-4/groups.png" width="400" height="300" />
</p>

***

#### Can we have both?
* DeepSeek asked can we get best of both worlds?
1. Low Perplexity (Low)
2. Low KV-Cache

#### January 2025
* Beautiful trick
* I don't want to do grouping across keys and values.
* Shift focus from reducing the nubmer of heads ($n_{heads}$) to compressing information within these heads.
* What if we don't have to cache K & V separately?
* What if: We could first project our input into a single, combined, much smaller matrix: a latent matrix and cache only that.
* Instead of caching two large matrics K and V, we only cache one smaller, lower dimensional matrix ($cKV$).
* The single matrix becomes our highly efficient cache.
* When we need full K & V cache, we can reconstruct them on the fly from the compressed latent representation.
* $C_{KV}$: Latent KV cache
* $W_{dKV}$: Latent down projection matrix

***

* 10:00

#### 7,168 -> 512
* **Note**: The embedding dimension ($d_{model}$) for major DeepSeek models like DeepSeek-V3, DeepSeek-R1, and DeepSeek-V4 is $7,168$
* Latent dimension : $512 = 2^{9}$

$$\text{Latent Matrix} = C_{KV}(N, d_{latent}) = X(N, d_{model}) \times W_{dKV}(d_{model}, d_{latent})$$

***

#### Formula for KV-Cache
```
l: layers
b: batch size
n: number of attn heads
s: seq length
h: head dim
2: K & V
2: byptes / parameter
```

$$l \times b \times \boxed{s} \times \underbrace{n_{heads} \times h}_{\text{embedding dim}} \times 2 \times 2$$

* Our foucs is now on sequence length (s).
* We have already seen one approach __DeepSeek Sparse Attention (DSA)__ to tackle this.

***

* 30:00

#### [Sliding Window Attention (SWA)](https://github.com/muarshad01/Inference-Engg/blob/main/SWA.md)

***

* 1:30:00

#### Convolution - Music
* Reverb - At each time point you add influences of that past.
* Reverb preset = $[a_1, a_2, a_3, a_4]$
* Time points   = $[x_1, x_2, x_3, x_4]$

#### Fourier Transform
* **Joseph Fourier (1807)**: Any signal can be decomposed into a sum of simple sine waves, each with its own frequence and amplitude.
* Every seq of number can be represented as sum of frequencies.
* $[3,5,3,1,3,5,3,1]$
* $$x_n=\frac{1}{8}\sum_{k=0}^{7}x_kW_{8}^{-kn}$$
* One $W_8=45^{o}$ angle

| Convolution ||
|---|---|
| Time Domain | $O(N^2)$ |
| Frequency Domain | $O(N \log N)$ |

***

* 2:00:00

#### CNN

* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [Demo Video "CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization"](https://www.youtube.com/watch?v=HnWIHWFbuUQ)

***

* 2:20:00

#### State Space Model (SSM)

$$
\begin{align}
   h_t &= A \times h_{t-1} + x_t\\
   y_t &= Ch_t\\
\end{align}
$$

***

* 2:30:00

#### [Linear Attention](https://github.com/muarshad01/Inference-Engg/blob/main/Linear-Attention.md)

* 3:00:00

#### [Mamba](https://github.com/muarshad01/Inference-Engg/blob/main/Mamba.md)

***

* 4:00:00

#### Axes of KV Compression: Heads and Tokens
* **Tokens**: Look at only W tokens in the past (Window=$W$)
* (A) sliding window
  * There is concept of **Effective Receptive Field (ERF)**
  * More number of layers L, the more broader is the ERF
  * KV cache size is reduced to $O(W)$ insted of $O(N)$ in MHA
* (B) **Detour 1**: convolution NlogN
* (C) **Detour 2**: SSMs are all around us
* (D) What if $\text{softmax}$ not there? -> Linear Attention -> Fixed KV size $(d,d)$ -> equal importance to ALL past.
  * Whenever a new token comes, the influence of past is compressed into a small $(d \times d)$ matrix.
  * The whole context is compressed into a small $(d \times d)$ matrix. $d^2$ has pressure to contain all information of past.
  * Context bottleneck problem (Same issue as in RNN).
  * Size of KV-Cache would reduce by a huge amount and stay fixed, but contxt is compressed into 
$(d \times d)$ values and as N increaes, it puts a lot of pressure on this small context to contain context of all past tokens.
* (E) Linear Attention (LA) is nothing but SSM
  * This is never used in practise
  * History is not weighted. For any new token gives same importance to past tokens.
* (F) Formalize Attn as SSM (use Detour 2):
  * Prefill: Convolution: $NlogN$ (use Detour 1)
  * Decode: fixed KV size
  * Decaying importance to past.
* (G) selective access to past
  * Mamba
  * Mamba prefill: Can't use Conv trick ($N\log N$) -> **Paralle Scan**
  * Mamba decode: Fixed KV size
  * Solve past token selectivity issue
* (H) Mamba -> Quality -> past is compressed into a small matrix
  * Can't be used standalone in practice. Have to be interleaved with traditional transformer block.
  * **NVIDIA - Nemotron 2 nano** uses Mamba Architecture
* (I) Alternate Mamba with full Attn layer
* (J) Good tradeoff -> Jamba

*** 

#### Research Topics
* MLA + Mamba
* Sliding window reception analysis for dynamic windows?
* Mamba + sliding window + full attention

***

| Method	| What is cached?	| Memory with context length N	| Intuition	| Likely trade-off | 
|---|---|---|---|---|
| __Full attention__	| Every token's K and V   | $O(N \times H_{KV} \times d)$ | 	Perfect address book of the past. | Strong recall, expensive long-context inference. | 
| __MQA / GQA__ | K and V for fewer KV heads	| $O(N \times \text{fewer heads} \times d)$ | Keep all tokens, reduce the head axis.	| Often excellent quality/speed balance. |
| __Sliding window__ | K and V for last W tokens in local layers	| $O(W \times H_{KV} \times d)$	 | Recent working memory. | Great local efficiency, weaker direct recall for old clues. |
| __Linear attention__	| Running numerator S and denominator z| 	$O(d_{\phi} × d_v)$	| Compress all past tokens into algebraic sums.	| Fast state updates, but softer token selection than softmax attention. | 
| __SSM / Mamba__	| Fixed SSM state plus small convolution buffer | 	O(state size), independent of N	| Compressed evolving memory.	| Huge memory savings; quality depends on how well the state preserves needed details. |
| __Hybrid models__	| Some attention cache plus some fixed state	| Depends on layer mix	| Use attention for exact lookup, SSM/local layers for efficiency.	| Often the practical middle path for long contexts. |

***

<p align="center">
<img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-4/grand-comparison.png" width="600" height="300" />
</p>

