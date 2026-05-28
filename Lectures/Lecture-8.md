## Speculative Decoding
* Inference:
  * Prefill (compute bound)
  * decode (memory bound)

***

* 15:00

#### Example: Memory Bound

|||
|---|---|
| Tiny Llama    | 1.1 B |
| Model Weights | 2.2 GB |
| HBM (A100)    | 2 TB/s |
|Time to download these wts to compute area | $\frac{2.2 ~GB}{2 ~TB/s} = 1.1 ~ms$ |
| FLOPs for 1 token decoding | 2 x num_params = 2.2 B|
| New Token: $x(1, m)$      |  |
| Weight query matrix: $W_q(m, n)$   ||
| FLOPs for $x(1, m) \times W_q(m, n)$   | 2 x mn |
| num_params $(W_q) = (m,n)$             | 2 x num_params |
| A200 | 312 TFLOPs |
| Time to decode one token | $\frac{2.2 B}{312 TFLOPs}=0.007~ms$|
| Ratio = 1.1ms (loading) / 0.007 ms (decoding) = 157 | GPU spends 99% time loading and only 1% time computing! |

***

* 30:00

* LLM decoding: autoregressive!

