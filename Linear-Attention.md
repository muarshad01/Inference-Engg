#### State Space 


* $N$: Number of tokens
* $d$: dimension size
* Cache Size = K and V matrices of Size Nd = $2 \times N \times d$

$$\text{softmax}\bigg(\frac{Q(N,d)\times K^T(d,N)}{\sqrt{d_{Keys}}}\bigg) \times V(N,d)$$

* What if $\text{softmax}$ is removed and we only have:

$$
\begin{align}
  & (Q \times K^T)\times V \\
  & Q \times \underbrace{(K^T \times V)}_\text{Cache this (dxd) matrix}\\
\end{align}
$$

* First multiply $K^T(d,N)\times V(N,d) \rightarrow A(d,d)$
* New token: $k_t^T(d,1) \times v_t(1,d) \rightarrow B(d,d)$
* Update cache: $A(d,d)+B(d,d)$

#### Observation
* Whenever a new token comes, the influence of past is compressed into a small $d \times d$ matrix.
* The whole context is compressed into a small $d \times d$ matrix.
* Context bottlenexk problem.
* Size of KV-Cache would reduce by a huge amount and stay fixed, but contxt is compressed into $d \times d$ values and as $N$ increaes, it puts a lot of pressure on this small context to contain context of all past tokens.

* [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

* [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

***

* 2:40:00

#### How can we magically remove softmax?
* Can we replace softmax with something which allows me to compute $(K^T \times V)$ first and still maintains the benefits of $\text{softmax}$?

#### Softmax Benefits
1. ALL positive values with $\sum_{i} = 1$
2. Normalized

$$\frac{[Q \times K^T] \times V}{Q \times K^T}$$

* Denominotor will ensure Normaliation.
* How can we ensure positive values?

***

#### Linear Attention

$$
\begin{align} 
   \text{Linear Attention} &= \frac{[\phi(Q)\times \phi(K^T)] \times V}{\phi(Q) \times \phi(K^T)}\\
                           &= \frac{\phi(Q)\times [\phi(K^T) \times V]}{\phi(Q) \times \phi(K^T)}\\
\end{align}
$$

* Linear $\text{softmax}$
* KV-Cache size = $d^2$

#### When a New Token Arrives
* $q_t[1,8] \rightarrow \phi(q_t) \rightarrow [1,8]$
* $k_t[1,8] \rightarrow \phi(k_t) \rightarrow [1,8]$
* $v_t[1,8] \rightarrow \phi(v_t) \rightarrow [1,8]$

 $$
\begin{align} 
   \text{Linear Attention} &= \frac{\phi(q_t)\times [\phi(k_t^T) \times v_t]}{\phi(q_t) \times \phi(k_t^T)}\\
                           &= \frac{\phi(q_t)\times \text{new cache (8,8)}}{\phi(q_t) \times \text{cache for } \phi(K_t^T)}\\
\end{align}
$$


#### Advantage
* $d^2$

#### Drawback of Linear Attention
* This is never used in practise
* History is not weighted. Gives same weitage to all older states.

***
* Cache (State) is maintained
* Cache is updated whenever a new Query (Input) comes in.
* LA is one of the simplest form of SSM.
***

* 2:45:00

#### Use a positive feature map: $\phi(X): \text{ELU}(x) + 1$

$$
  \text{ELU}(x)=
  \begin{cases}
    x+1, & \text{if $x>0$}.\\
    e^{x}, & \text{if $x<0$}.
  \end{cases}
$$

***


#### FLOPs during Prefill
* Queries: $N$ Queries of dimension $d$
* Keys: $N$ Keys of dimension $d$

|   | Prefill FLOPs | Decode FLOPs| |
|---|---|---|---|
| MHA | $2 \times N_{queries} \times N_{queries} \times d$ | $2 \times N \times d$ | $N_{queries}=1$ |
| SWA | $2 \times N_{queries} \times W_{keys} \times d$ | $2 \times W \times d$ |$N_{queries}=1$|
|---|---|---|---|
| Linear Attention | $N\times d^2 + N \times d$ | ||
