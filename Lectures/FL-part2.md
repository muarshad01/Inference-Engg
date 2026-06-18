1. Disaggregated serving (Prefill and Decode)
2. Building a million user production interface system

***

#### Chunked Prefill in vLLM
* Users: A,B,C,D
* (A,B,C) decoding - i.e., 1-token each
* D prefill - say like 12,000 tokens

#### Disaggregated serving 
* Prefill pool
* KV-cache transfer
* Decode pool

***

* 15:00


