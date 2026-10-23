## Lecture-2: KV Cache
* __Need__ of KV Cache
* __Benefits__ of KV Cache
* __Dark side__ of KV Cache

***

#### Companies
* [Companies](https://github.com/muarshad01/Inference-Engg/blob/main/companies.md)

*** 

<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-2/LLM.png" width="400" height="300" />
</p>

* Assume embedding dimension = 8

| Single HA | Multi HA ||
|---|---|---|
| $X(4,8)=X(s,d)$ | $X(4,8)=X(s,d)$ ||
|---|---|---|
| Trainable $W_Q (8,4)=W_Q (d_{in},d_{out})$ | $W_{Q_1} (8,2)$ <br> $W_{Q_2} (8,2)$ | $$d_{head} = \frac{d_{out}}{n_{head}} = \frac{4}{2} = 2$$|
| Trainable $W_K (8,4)$ | $W_{K_1} (8,2)$ <br> $W_{K_2} (8,2)$ ||
| Trainable $W_V (8,4)$ | $W_{V_1} (8,2)$ <br> $W_{V_2} (8,2)$ ||
|---|---|---|
| $Q(4,4)=X \times W_Q$ | $Q_1(4,2)=X \times W_{Q_1}$ <br> $Q_2(4,2)=X \times W_{Q_2}$||
| $K(4,4)=X \times W_K$ | $K_1(4,2)=X \times W_{K_1}$ <br> $K_2(4,2)=X \times W_{K_2}$||
| $V(4,4)=X \times W_V$ | $V_1(4,2)=X \times W_{V_1}$ <br> $V_2(4,2)=X \times W_{V_2}$||
|---|---|---|
| $A(4,4)=Q \times K^T$ | $A_1(4,4)=Q_1 \times K_1^T$ <br> $A_2(4,4)=Q_2 \times K_2^T$ | - Attention score matrix <br> - Size of the attention socre matrices remains exactly the same. <br> - Number of attention socre matrices is same as number of attention heads.|
| $B(4,4)=\text{softmax}\bigg(\text{casual}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg)$ | $B_1(4,4)$ <br> $B_2(4,4)$ | Attention weight matrix |
| $Z(4,4)=\text{softmax}\bigg(\text{casual}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg) \times V$ | $Z_1(4,2)=\text{Perspective-1}$ <br> $Z_2(4,2)=\text{Perspective-2}$ | Context matrix |
|| $Z(4,4) = [Z_1(4,2) \parallel Z_2(4,2)]$ |Concatinate Context matrices|

***

#### Why divide by $\sqrt{d_{keys}}$ and Why apply `softmax`
* [softmax](https://github.com/muarshad01/DeepSeek/blob/main/softmax.md)

***

* 30:00

#### Perspective

```
The artist painted the portrait of a woman with a brush.
```

1. painting a woman with a "brush" (in artists other hand.)
2. painting of a "woman with a brush"

*** 

#### The problem of redundant computations
* Output: (num_tokens, vocab_size)
   * Only the last row is relevant to predict the next token
* Are we calculating the same thing over and over

***

* 1:00:00

#### KV Cache

<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-2/Decode-1.png" width="400" height="300" />
</p>

* Prefill = N
* Decode = N
* Attention Score = $O(N^2)$
* For each of N Decode, we have to perform $O(N^2)$ operations = Total operations $O(N^3)$

***

* Basically, we only need query vector, key vector, and value vector for "bright".
* $X_{bright}(1,8) \times W_k(8,4)$ = Key vector for bright (1,4) -> Append to keys cache
* $X_{bright}(1,8) \times W_q(8,4)$ = Query vector for bright (1,4)
* $X_{bright}(1,8) \times W_v(8,4)$ = Value vector for bright (1,4) -> Append to values cache

***

#### Inference loop with KV Cache
* Receive new token: The model looks up the embedding for new token ($X_{bright}(1,8)$).
* Compute Query vector, Key vector, and Value vector
* Retrieve from cache: K and V cache of earlier tokens
* Append to cache: Append Key and Value vectors of new token to existing cache
* Multiply Query vector with keys -> Attention scores -> Attention weights
* Multiply Attention weights with Values -> Context vector for new token
* Context vector * W_0 -> get out of my attention block
* FFN
* Multiple transformer blocks
* Logits vector for new tokens
* select next token

***

* 1:30:00

|||||||
|---|---|---|---|---|---|
| __Pre-training__ | Params ($W_Q, W_K, W_V$) | Grads  | Optims  | Activs ($Q, K, V, A, softmax(A), Z$) | ---|
| __Inference__    | Params ($W_Q, W_K, W_V$) | ---    | ---     | Activs ($Q, K, V, A, softmax(A), Z$) | KV-cache |

* 1:40:00

***

#### Good Side: Necessary Evil
* Benefit: Saves recomputation
  * Without KV-Cache, we're calculating the same thing over and over (i.e., redundant computation)?
* During Inference KV-cache Benefits (TTFT, ITL, TPS)
* KV-cache benefits ITL, becauses we're doing less FLOPs
  * $\text{num FLOPs} \propto \text{Wall clock time}$
  * $\text{num FLOPs} (\downarrow) \longrightarrow  \text{Wall clock time} (\downarrow) \longrightarrow ITL(\downarrow)$
  * $\text{num FLOPs} (\uparrow)   \longrightarrow  \text{Wall clock time} (\uparrow)   \longrightarrow ITL(\uparrow)$

***

* Saves us number of FLOPs, which speedsUp Inference.
* Number of FLOPs saved goes on increasing with $\text{SeqLength}$
* $\text{SeqLength} (\uparrow) \longrightarrow SpeedUp(\uparrow)$

*** 

#### Dark Side: Size of KV Cache

<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-2/KV-cache-size.png" width="400" height="300" />
</p>

* l : number of transformer blocks layers
* b : batch size
* h : attention head dimension ($d_{head}$)
* $n_{heads}$ : number of attention heads
* s : sequence length (context length)
* 2 : number of byptes per Parameter (Assume each parameter takes 2 bytes)
* 2 : Two caches one each for (k,v)

$$\bigg(\text{Bytes ~taken ~up ~by ~KV cache}\bigg) = l \times b \times \underbrace{s \times h \times n_{heads}}_\text{One Layer} \times 2 \times 2$$

***

<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-2/kv-cache-dark-side.png" width="400" height="300" />
</p>

* High Bandwidth Memory (HBM) needs to be transferred to compute.
* In the roofline graph, X-axis is called Arithmetic Intensity (AI), which is (FLOPs/bytes)__.
* As the number of bytes increase, we shift towards left/down along the slope, i.e., become more MEMORY Bound.

<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-2/Llama3-memory-budget.png" width="400" height="300" />
</p>

***

* 2:05:00

#### Approaches to Reduce Size of KV-Cache
1. Quantization (e.g., FP16 -> FP8; store less bytes)
2. Compress: Projection into Latent Space (MLA)
3. Temporal Compression (DeepSeek V4): past tokens (high compression) and recent past (low compression); 
4. Reduce $n_{heads}$ (MQA, GQA)
5. Limit the SeqLength(s) (Sliding WA)
6. State Space Models (Mamba, Jamba)

***

* 2:10:00

#### RunPod
```
GPU: H100 SXM 
Pod name: KV Cache Inference Worskshop
Volume Disk
Deoploy
```

<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-2/seqlen-speedup.png" width="400" height="300" />
</p>

* $\text{SeqLength} (\uparrow) \longrightarrow SpeedUp(\uparrow)$
* $\text{num FLOPs} (\downarrow)   \longrightarrow  \text{Wall clock time} (\downarrow)   \longrightarrow ITL(\downarrow)$
* Number of FLOPs saved goes on increasing with $\text{SeqLength}$


<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-2/seqlen-users.png" width="400" height="300" />
</p>

***

* 2:35:00

#### TurboQuant

| Paper ||
|---|---|
| [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | Can we store KV-cache in low precisin. |
| [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | Temporal Caching for KV Cache |
| [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)](https://arxiv.org/abs/2208.07339) ||
| [DeepSeek V4 in vLLM: Efficient Long-context Attention](https://vllm-project.github.io/2026/04/24/deepseek-v4.html) ||

* Keys dimesion = 768
* Values dimension = 768 (The call it as channels)
* Outlier problem across dimensions (some dimensions have really high values.)

***

* First convert the vector so that it remains on a unit sphere.
* Secondly, rotate that on sphere so that spikyness reduces.

***

* Spiky Vector (V_1) -> Quantize (V_2) -> De-quantize (V_3)
* L_1 = Loss(V_1, V_3)
* Spiky Vector (V_1) -> Rotate -> Quantize (V_2) -> De-quantize (V_3^')
* L_2 = Loss(V_1, V_3^')
* L_2 <<< L_1

***

#### REDO This part

* Weights are quantized in blocks
* Activations are quantized in channels (Spikyness issue only showsup in Activations)

***

* 3:20:00

#### DeepSeek V4 KV-Cache: Temporal Caching
1. near past
2. middle past (compressed version of tokens - C4A)
3. far past (compressed version of tokens - C128A)

***

* 3:30:00

* Why are we not using TurboQuant
* Why are we not implementing DeepSeek V4 KV-Cache in our OpenSource LLM

#### Research Paper Idea
1. Start with an OpenSource model
2. Ask Claude to Digest everything from DeepSeek V4 paper
3. Implement the V4 paper changes on the Qwen model
4. Implement TurboQuant
5. Then compare and constast the two
6. Then compare

#### Research Paper Idea
* TurboQuant with LoRA
* Coding is not the bottleneck; Ideas is the bottleneck!!!
* Collaborate using Claude code.

****
