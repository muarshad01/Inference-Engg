## Lecture-5

#### All about Flash Attention

#### Review
* Sliding Window: We can only look $W$ tokens in the past
* There is concept of __Receptive Field__
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
* $Q(16,8)$
* $K(16,8)$
* $V(16,8)$

* Tile size= 4 Tokens
* $Q_1(4,8)$
* $K_1(4,8)$
* $V_1(4,8)$

* $A = Q_1 \times K_1^T$
* $O = \bigg( \frac{\exp^{s_1}}{Sum_1}, \bigg)$
