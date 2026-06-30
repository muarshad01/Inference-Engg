## Lecture-4

#### Four Paradigmsn for Compressing Attention Across Tokens
* Full Attention (baseline)
* [Sliding Window Attention (SWA)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/06_swa/README.md)
* Linear Attention 
* State Space Models (SSM) - Mamba Architecture

***

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-4/4-paradigms-attentions.png" width="500" height="400" />
</p>


***

* 25:00

$$l \times b \times \boxed{s} \times \underbrace{n_{heads} \times h}_{\text{embedding dim}} \times 2 \times bytes$$

* Our foucs is now on sequence length (s).
* We have already seen one approach __DeepSeek Sparse Attention (DSA)__ to tackle this.

***

* 30:00

#### [Sliding Window Attention](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/06_swa/README.md)
* N: Sequence length
* W: Sliding window length
* We reduce KV-cache size by a factor of $\frac{W}{N}$

#### Trade-offs
* Long range dependency is completely lost.
* No index, which is bad! (We don't know which tokens are important.)
* Lazy compared to DSA.
* Attend to unimportant tokens. Miss out on important tokens.

* How researchers mitigated drawback of SWA?

***

#### Gemma
* $\to TE \to PE \to T_1(Sliding) \to T_2(Sliding) \to T_3(Sliding) \to T_4(Sliding) \to T_5(Full) \to T_6(Sliding) \to T_7(Sliding) \to T_8(Sliding) \to T_9(Sliding) \to T_{10}(Full) \to T_{11} \to T_{12} \ldots T_{96} \to Logits \to NextToken$

***

* 40:00

#### Effective Receptive Field
* L : num Layers
* W : Window Size
* $W \times L$

* 50:00

***

#### Active Research Area
* Receptive Field of Attention Mechanaism (Promising area of research)
* Dynamic sliding window sizes ($$W=f(L)$$)

***

* 1:20:00

#### FLOPs during Prefill
* Queries: $N$ Queries of dimension $d$
* Keys: $N$ Keys of dimension $d$

| Reduction |   | Prefill (Compute Bound Regime) | Decode (Memory Bound Regime) | 
|---|---|---|---|
|               | MHA              | $2 \times N^2 \times d$        | $2 \times N \times d$ |
| $\frac{W}{N}$ | SWA              | $2 \times N \times W \times d$ | $2 \times W \times d$ |
|               | Linear Attention | $N\times d^2 + N \times d$ | |
|               | SSM              | $N \log N$ | $d^2$ | 

***

#### Convolution
* FFT: $O(N^2) \to O(N \log N)$

***

* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [Demo Video "CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization"
](https://www.youtube.com/watch?v=HnWIHWFbuUQ)

***

* 2:20:00

#### State Space Model (SSM)

$$h_t = A \times h_{t-1} + x_t$$

***

* 2:30:00

* $Q(N,d)$
* $K(N,d)$
* $V(N,d)$

$$\text{softmax}\bigg(\frac{Q(N,d)\times K^T(d,N)}{\sqrt{d_{Keys}}}\bigg) \times V(N,d)$$

* Cache Size = $2 \times N \times d$

* What if $\text{softmax}$ is removed and we only have $(Q \times K^T) \times V$.
* We can first multiply $A(d,d) = K^T(d,N)\times V(N,d)$
* New token: $B(d,d) = k_t^T(d,1) \times v_t(1,d)$
* Update cache: $A(d,d)+B(d,d)$
* __Observation__:
* Whenever a new token comes, the influence of past is compressed into a small $d \times d$ matrix.
* The whole context is compressed into a small $d \times d$ matrix.
* Context bottlenexk problem.
* Size of KV-Cache would reduce by a huge amount and stay fixed, but contxt is compressed into $d \times d$ values and as $N$ increaes, it puts a lot of pressure on this small context to all past tokens.

* [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

* [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

***

* 2:40:00

#### How can we magically remove softmax?
* Can we replace softmax with something which allows me to compute $K^T \times V$ first and still maintains the benefits of softmax?

#### Softmax benefits
1. ALL positive values with $\sum_{i} 1$
2. Normalized

$$\frac{[Q \times K^T] \times V}{Q \times K^T}$$

* Denominotor will ensure Normaliation.
* How can we ensure positive values

#### Linear Attention

 $$
\begin{align} 
   \text{Linear Attention} &= \frac{[\phi(Q)\times \phi(K^T)] \times V}{\phi(Q) \times \phi(K^T)}\\
                           &= \frac{\phi(Q)\times [\phi(K^T) \times V]}{\phi(Q) \times \phi(K^T)}\\
\end{align}
$$

* Linear \text{softmax}
* ~ KV-Cache size = $d^2$

#### New Token

 $$
\begin{align} 
   \text{Linear Attention} &= \frac{\phi(Q)\times [\phi(K^T) \times V]}{\phi(Q) \times \phi(K^T)}\\
                           &= \frac{\phi(Q)\times \text{cache}}{\phi(Q) \times \text{cache for } \phi(K^T)}\\
\end{align}
$$


#### Advantage
* $d^2$

#### Drawback of Linear Attention
* This is never used in practise
* History is not weighted???

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

* 3:00:00

### State Space Models (SSM) for Attention

* $x_t \to \text{MHA}(h_t) \to y_t$

$$
\begin{align}
   h_t &= Ah_{t-1} + Bx_t\\
   y_t &= Ch_t\\
\end{align}
$$

***

$$
\begin{align}
   h_t &= Ah_{t-1} + Bx_t\\
   h_0 &=    Bx_0 \\
   h_1 &=   ABx_0 +    Bx_1\\
   h_2 &= A^2Bx_0 +   ABx_1 + Bx_2\\
   h_3 &= A^3Bx_0 + A^2Bx_1 + ABx_2 + Bx_3\\
\end{align}
$$

***

$$
\begin{align}
   y_t &= Ch_t\\
   y_0 &=    CBx_0 \\
   y_1 &=   CABx_0 +    CBx_1\\
   y_2 &= CA^2Bx_0 +   CABx_1 + CBx_2\\
   y_3 &= CA^3Bx_0 + CA^2Bx_1 + CABx_2 + CBx_3\\
\end{align}
$$

****

$$
\begin{align}
   y_0 &= k_0x_0 \\
   y_1 &= k_1x_0 + k_0x_1\\
   y_2 &= k_2x_0 + k_1x_1 + k_0x_0\\
   y_3 &= k_3x_0 + k_2x_1 + k_1x_2 + k_0x_3\\
\end{align}
$$

***

* 3:20:00
  
* Although this is better an Linear Attention: all past does not have same weight.
* Past contribution decays esponentially.
* But A and B are still fixed!
* Not good!
* I need to have some selectivity!
* I need to vary A and B also for every new token which comes in.

***

#### Augment SSM with Selectivity

$$
\begin{align}
   h_t &= A_th_{t-1} + B_tx_t\\
   y_t &= C_th_t\\
\end{align}
$$

* $A_t$: How much of the previous hidden state ($h_t$),  we needs to pay attention to 
* $B_t$: How much of the current token, we need to pay attention to
* $C_t$: The output factor

$$
\begin{align}
   \hat{B}_t    &= \Delta_t   \times B_t \\
   \hat{A}_t &= \exp(\Delta_t \times A_t)\\
\end{align}
$$

* $B_t$: How much of the current token, we need to pay attention to
  * If $\Delta_t$ is high, the current input will get lot of weitage
* $A_t$: How much of the previous hidden state ($h_t$),  we needs to pay attention to 
  * If $\Delta_t$ is high then A will be negative, i.e. $\hat{A}_t$ will be low. That is past will not get more weitage.
* $C_t$: The output factor

***

* 3:25:00

#### Selective SSM --> Mamba Architecture
* Mamba Block (Conv1D + SSM + Gate)

***

* 3:50:00

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

