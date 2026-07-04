#### [Sliding Window Attention](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/06_swa/README.md)
* N: Seq length
* W: Sliding window length (We consider a fixed number of past tokens.)
* We reduce KV-cache size by a factor of $\frac{N}{W}$
* DSA: Which tokens of past are important.

#### Trade-offs
* Long-range dependency is completely lost.
* No Indexer, which is bad. We don't know which tokens are important!
* Lazy compared to DSA
* Attend to unimportant tokens and miss out on important tokens.

***

#### [Gemma](https://sebastianraschka.com/llm-architecture-gallery/)
* **Question**: How researchers mitigated drawback of SWA?
* $\to TE \to PE \to T_1(Sliding) \to T_2(Sliding) \to T_3(Sliding) \to T_4(Sliding) \to T_5(Full) \to T_6(Sliding) \to T_7(Sliding) \to T_8(Sliding) \to T_9(Sliding) \to T_{10}(Full) \to T_{11} \to T_{12} \ldots T_{96} \to Logits \to NextToken$

***

* 40:00

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-4/receptive-field-growth.png" width="500" height="400" />
</p>

#### Effective Receptive Field
* L : num Layers
* W : Window Size
* $W \times L$

***

* 50:00

<p align="center">
   <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-4/variable-window.png" width="400" height="400" />
</p>


#### Receptive Field (Max depth we can reach) of Attention Mechanaism 
* Promising area of research
* Dynamic sliding window sizes $W=f(L)$

#### MHA versus SWA
* MHA - Direct memory
* SWA - Fading memory of past
* How do we guarantee more fading memory?
* Increase #Layers
* $O(L \times W)$

***

* 1:20:00

#### FLOPs during Prefill
* Queries: $N$ Queries of dimension $d$
* Keys: $N$ Keys of dimension $d$

|   | Prefill FLOPs | Decode FLOPs| |
|---|---|---|---|
| MHA | $2 \times N_{queries} \times N_{queries} \times d$ | $2 \times N \times d$ | $N_{queries}=1$ |
| SWA | $2 \times N_{queries} \times W_{keys} \times d$ | $2 \times W \times d$ |$N_{queries}=1$|
|---|---|---|---|
| Linear Attention | $N\times d^2 + N \times d$ | ||

* We reduce KV-cache size by a factor of $\frac{N}{W}$ for SWA compared to MHA.


***
