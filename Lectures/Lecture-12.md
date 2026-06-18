


#### Capstone Project 1 (SGLang + Megatron + Ray + Slime)

* OpenClaw-RL — Self-Improving WhatsApp AI Assistant__ — A full RL pipeline where your everyday messages become training data. Build and deploy a personal AI assistant that improves from every conversation using reinforcement learning — no labeling, no datasets.

* OpenClaw RL, from the Inference Side

1. Whey **personalization** needs learning, not memory
2. How the inference and training clusters interact
3. Where the optimization leavers are
4. What RL means to OpenClaw setting
5. First-principles inference clost

***

* RL needs multiple copies of same LLM
* Pre-training / Post-training (Inference comes here!)

* Goal: Build a personal assistant (PA) that listents to us and modies itselfe based on our interaction with PA.

1. Whey **personalization** OpenClawn RL
2. 

***

* 10:00

* Memory-based personalization
* Learning-based personalization

*** 

* RAG: Indexing at run time
* Indexing at design time

* [PageIndex: Vectorless, Reasoning-based RAG](https://github.com/VectifyAI/PageIndex)

*** 

* 20:00

#### RL

|Game of Chess||
|---|---|
| state | observation|
| action ||
| reward | Long-term success|

#### 2018
* Games
* Robotics
* LLms

#### LLM
* LLM: Next predicted token becomes action.

$$\underbrace{The ~sky ~is}_{state} ~\underbrace{blue.}_{action}$$

***

* 30:00

#### Reward Model
* Is also a LLM
* Outcome based reward
* Process based reward

***

* 40:00

* 2020 - RLHF - ChatGPT - 
* 2024 - RLAIF - LLM as a judge

***

#### PPO
* Proximal Policy Optimization (PPO) - 2017 - OpenAI
1. LLM
2. Reward Model
3. Value Model (Critic -> Gives Average)
4. Reference Model
* Clipping: KL (LLM || Ref Model)

***

* 50:00

#### Group Reletive Policy Optimizaiton (GRPO)
* DeepSeek - 2025
1. LLM
2. Reward Model (RLRM)
3. ~~Value Model (Critic -> Gives Average)~~
4. Reference Model

***

* 1:30:00

* [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35)
* [SGLang](https://github.com/sgl-project/sglang) Judge
* [Megatron-LM](https://github.com/nvidia/megatron-lm)
* [slime - GRPO](https://github.com/THUDM/slime) - slime is not used if no RL!

***
