### State Space Models (SSM) for Attention

* $x_t \to \text{MHA}(h_t) \to y_t$

$$
\begin{align}
   h_t &= Ah_{t-1} + Bx_t\\
   y_t &= Ch_t\\
\end{align}
$$

* Prefill: $$O(N\log N)$$ FLOPS
* Decode: Const cache size as $h_t$ only needs $h_{t-1}$

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
   y_2 &= k_2x_0 + k_1x_1 + k_0x_2\\
   y_3 &= k_3x_0 + k_2x_1 + k_1x_2 + k_0x_3\\
\end{align}
$$

***

* 3:20:00
  
* Although this is better than Linear Attention: all past does not have same weight.
* Past contribution decays esponentially.
* But A and B are still fixed, which it NOT good!
* I need to have some selectivity!
* I need to vary A and B also for every NEW token, which comes in.

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
* SiLU activation function.

***
