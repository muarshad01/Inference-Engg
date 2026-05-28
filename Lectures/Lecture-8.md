## Speculative Decoding
* Inference:
  * Prefill (compute bound)
  * decode (memory bound)

***

* 15:00

#### Example: Memory Bound

|||
|---|---|
| Tiny Llama    | 1.1B |
| Model Weights | 2.2 GB |
| HBM (A100)    | 2 TB/s |
|Time to download these wts to compute area | $\frac{2.2 ~GB}{2 ~TB/s} = 1.1 ~ms$ |
| FLOPs for 1 token decoding | 2 x Params |
| $x(1, m)$     ||
| $W_q(m, n)$   ||
| FLOPs for $x(1, m) \times W_q(m, n)$   | $2 \times m \times n$|
|       $Parameters (W_q) = (m,n)$             | $2 \times Parametres$|

***

* 30:00
