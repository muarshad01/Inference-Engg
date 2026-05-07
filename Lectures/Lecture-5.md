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

#### Attentions
1. $Q, K, V$ matrices are returned to HBM
2. Read $Q, K$ from HBM and compute attention score $A(N,N) = Q \times K^T$
3. Write $A(N, N)$ attention scores $\to$ HBM
4. Read attention score $A(N, N)$ from HBM and comptue $s=\text{softmax}(A)$
5. Write $\{softmax}(s)$ to HBM
6. Read $s$ and $V$ from HBM and compute $s\times V \to \text{context}$ 

* 3r + 3w
