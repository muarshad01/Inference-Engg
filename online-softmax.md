#### Main Idea
* Tile Q,K,V
* Bring tiles into SRAM
* Find partial output vectors
* Merge
* Send output vectors from SRAM to HBM.

***

* 1:15:00 

* [Online Softmax - Day08](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-08.md)

* $Q(16,8) - \\{q_1,q_2, ..., q_{16}\\}$
* $K(16,8) - \\{k_1,k_2, ..., k_{16}\\}$
* $V(16,8) - \\{v_1,v_2, ..., v_{16}\\}$

$$
\begin{align}
   A &= Q\times K^T\\
   S &= \text{softmax}(A)\\
   O &= S \times V
\end{align}
$$

* $q_1 (1,8) \times K^T(8,16) \rightarrow \\{s_1,s_2, ..., s_{16}\\} \rightarrow \\{\{\frac{e^{s_1}}{\text{sum}}, \frac{e^{s_2}}{\text{sum}}, ..., \frac{e^{s_{16}}}{\text{sum}}\}\\}$
* $\\{\{\frac{e^{s_1}}{\text{sum}}, \frac{e^{s_2}}{\text{sum}}, ..., \frac{e^{s_{16}}}{\text{sum}}\}\\} \times V(16,8) \rightarrow O(1,8)$ 

***

* Consider Tile size= 4 Tokens

| Q | K | Attention Score | Attention Weights | Context Vector |  Store |
|---|---|---|---|---|---|
| $q_1(1,8)$ | $K_1(4,8)$ | $A_1 = q_1 \times K_1^T$  | $S_1 = \bigg( \frac{e^{s_1}}{sum_1}, \frac{e^{s_2}}{sum_1}, \frac{e^{s_3}}{sum_1}, \frac{e^{s_4}}{sum_1}\bigg)$   | $O_1 = S_1 \times V_1(4,8)$ | $O_1 \times sum_1$; running-sum |
| $q_2(1,8)$ | $K_2(4,8)$ | $A_2 = q_2 \times K_2^T$  | $S_2 = \bigg( \frac{e^{s_5}}{sum_2}, \frac{e^{s_6}}{sum_2}, \frac{e^{s_7}}{sum_2}, \frac{e^{s_8}}{sum_2}\bigg)$   | $O_2 = S_2 \times V_2(4,8)$ | $O_1 \times sum_1 + O_2 \times sum_2$; running-sum |
| $q_3(1,8)$ | $K_3(4,8)$ | $A_3 = q_3 \times K_3^T$  | $S_3 = \bigg( \frac{e^{s_9}}{sum_3}, \frac{e^{s_{10}}}{sum_3}, \frac{e^{s_{11}}}{sum_3}, \frac{e^{s_{12}}}{sum_3}\bigg)$ | $O_3 = S_3 \times V_3(4,8)$ | $O_1 \times sum_1 + O_2 \times sum_2 + O_3 \times sum_3$; running-sum |
| $q_4(1,8)$ | $K_4(4,8)$ | $A_4 = q_4 \times K_4^T$  | $S_4 = \bigg( \frac{e^{s_{13}}}{sum_4}, \frac{e^{s_{14}}}{sum_4}, \frac{e^{s_{15}}}{sum_4}, \frac{e^{s_{16}}}{sum_4}\bigg)$  | $O_4 = S_4 \times V_4(4,8)$ | $O_1 \times sum_1 + O_2 \times sum_2 + O_3 \times sum_3 + O_4 \times sum_4$; running-sum|

* After each iteration, we tore $O_1$ and running sum of den value.

***

#### SRAM Memory (M) Size
* $M_{SRAM} = d \times \text{tile size}$

$$
\begin{align}
   \text{How many query tiles} &= \frac{N}{Tile ~size} =  \frac{N}{\frac{M}{d}} = \frac{Nd}{M}\\
\end{align}
$$

* **Standard Attention**:
   * For $Q,K,V$ matricies of dimension $(N,d)$ - $O(3Nd)$ R/W
   * For $A = Q\times K^T$ and $S = \text{softmax}(A)$ matrices of dimension $(N,N)$ - $O(2N^2)$
* **Flash Attention**:
   * For each query tile, I need to fetch all K and V from HRAM.
      * K and V size is $Nd$
   * Number of query tiles = $\frac{N}{\text{tile size}}$
      * $Nd \times \frac{N}{\text{tile size}}$
      * $Nd \times \frac{N}{\frac{M}{d}}=\frac{N^2d^2}{M}$

| Scheme |  | R/W from HRAM ||
|---|---|---|---|
| Standard Attention | $3Nd + 2N^2$ | $Nd + N^2$ ||
| Flash Attention    |              | $Nd + \frac{N^2d^2}{M}$ | $\frac{d^2}{M} <<< 1$ |

***

* 2:20:00

#### Variants of Flash Attention

* [Dao AI Lab](https://github.com/Dao-AILab)

| Paper ||
|---|---|
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) ||
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) ||
| [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) | - Specilized hardware: Tensor memory accelerator. <br> - Producer Consumer Pipeline|
| [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451) ||

***

* 2:40:00

#### Code
* H100 or H200
* PyTorch SDPA - Scaled Dot Product Attention

***

* 3:00:00

***
