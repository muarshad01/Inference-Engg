## Flash Attention 1,2,3
* Cornerstone of both inference and per-traing

#### Axes of KV Compression: Heads and Tokens
* **Tokens**:
* (A) Sliding window
* (B) Detour 1: Convolution NlogN
* (C) Detour 2: SSM are all around us



#### Review
* Sliding Window: We can only look $W$ tokens in the past
* There is concept of __Effective Receptive Field (ERF)__
  * More number of layers $L$, the more broader is the receptive field
* Active research area: Receptive field
* NVIDIA Nemotron uses Mamba Architecture

***

* 40:00

| Stage | Stored in Memory|
|---|---|
| Pre-Training | Params, Grads, Optims, Activs |
| Inference    | Params, Activs, KV-cache |

***

* 1:00:00

***

* High Bandwidth Memory (HBM)

#### Total RW = 3r + 3w
* Attention Block
1. Activations: $Q, K, V$ matrices are written to HBM
2. Read $Q, K$ from HBM and compute attention score, $A(N,N) = Q \times K^T$
3. Write matrix A $\to$ HBM
4. Read A from HBM and comptue $S=\text{softmax}(A)$
5. Write $S=\{softmax}(A)$ to HBM
6. Read $S$ and $V$ from HBM and compute $S \times V \to \text{context}$ 
* 3r + 3w
* __Question__: $N \times N$ matrix moves through GPU how many times? -- 2 times

***

#### Problem with Traditional Attention

```
N=4,096
4,096 x 4,096 x 2 =
2^{12} x 2^{12} x 2^{1} = 2^{25} = 32 MB for One Layer
One Layer --> 32 MB
32 Layers --> 1 GB / forward pass -- just for attention
* This whole 1 GB has to pass through HBM!
```

***

* You near write $N \times N$ matrices into HRAM
* Break it into smaller tiles, which briefly exist in SRAM

***

#### Idea
* Tile Q,K,V
* Bring tiles into SRAM
* Find partial output vectors
* Merge
* Send output vectors from SRAM to HBM.

***

* 1:30:00

***  

* [Online Softmax - Day08](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-08.md)

* $Q(16,8)$
* $K(16,8)$
* $V(16,8)$

* Tile size= 4 Tokens

| Q | K | | Attention Score | Context Vector |  |
|---|---|---|---|---|---|
| $Q_1(4,8)$ | $K_1(4,8)$ | $A_1 = Q_1 \times K_1^T$  | $S_1 = \bigg( \frac{e^{s_1}}{sum_1}, \frac{e^{s_2}}{sum_1}, \frac{e^{s_3}}{sum_1}, \frac{e^{s_4}}{sum_1}\bigg)$   | $O_1 = S_1 \times V_1(4,8)$ | $(O_1 \times sum_1, cum-sum)$ |
| $Q_2(4,8)$ | $K_2(4,8)$ | $A_2 = Q_2 \times K_2^T$  | $S_2 = \bigg( \frac{e^{s_5}}{sum_2}, \frac{e^{s_6}}{sum_2}, \frac{e^{s_7}}{sum_2}, \frac{e^{s_8}}{sum_2}\bigg)$   | $O_2 = S_2 \times V_2(4,8)$ | $(O_1 \times sum_1 + O_2 \times sum_2, cum-sum)$ |
| $Q_3(4,8)$ | $K_3(4,8)$ | $A_3 = Q_3 \times K_3^T$  | $S_3 = \bigg( \frac{e^{s_9}}{sum_3}, \frac{e^{s_{10}}}{sum_3}, \frac{e^{s_{11}}}{sum_3}, \frac{e^{s_{12}}}{sum_3}\bigg)$ | $O_3 = S_3 \times V_3(4,8)$ | $(O_1 \times sum_1 + O_2 \times sum_2 + O_3 \times sum_3, cum-sum)$|
| $Q_4(4,8)$ | $K_4(4,8)$ | $A_4 = Q_4 \times K_4^T$  | $S_4 = \bigg( \frac{e^{s_{13}}}{sum_4}, \frac{e^{s_{14}}}{sum_4}, \frac{e^{s_{15}}}{sum_4}, \frac{e^{s_{16}}}{sum_4}\bigg)$  | $O_4 = S_4 \times V_3(4,8)$ |$\bigg(\frac{O_1 \times sum_1 + O_2 \times sum_2 + O_3 \times sum_3 + O_4 \times sum_4}{sum_1 + sum_2 + sum_3 + sum_4}\bigg)$|

***

* 1:50:00

***

* $\text{Memory (SRAM)} = d \times \text{tile size}$


||||
|---|---|---|
| Tradition | $Nd + N^2$ | |
| Flash     | $Nd + \frac{N^2d^2}{M}$ | $\frac{d^2}{M} << 1$ | 

*** 

* Number of R/W per query time = $Nd$
* How many query tiles = $\frac{N}{Tile size} = \frac{N}{\frac{M}{d}} = \frac{Nd}{M}$
* $M = d \times \text{Tile size}$

***

* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
* [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)
* [Dao AI Lab](https://github.com/Dao-AILab)

***

* 2:55:00


