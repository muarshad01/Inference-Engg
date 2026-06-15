* Dr. Sreedath Panat - MIT PhD | IITM

#### Multimodel Inference
1. ViT - Vision Transformer
2. VLM - Vision Language Model
3. Image and video generation (using diffusion)
4. Video Gen

***

* SigLIP (Vision Laguage Model)
* Qwen2-VL (Vision Laguage Model)
* Flux (Image generation)
* Wan2.1 (Video generation)

***

#### 1. ViT - Vision Transformer

| Paper |
|---|
| [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - (June 2021)](https://arxiv.org/abs/2010.11929) |

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-11/vit_patch_embedding.png" width="500" height="300" />
</p>

* **Key point**: We need a tokenizer for pixels. ViT calls it the patch embedding.
* **Image**: Divide into Patches
* **Patch**: Group of pixels
* **Patch size**: $(16 \times 16)$ = 256 pixels
* 1 token = 1 Patch
* Number of tokens = Number of Patches
* **Flattening**: 2D-Conv operation for flattening of patches

$$
\frac{(224 \times 224)}{1 ~token:(16 \times 16)}=(14 \times 14)=196 ~patches= 196 ~tokens
$$

* **Special Token**: CLS or Classification token

#### DataSet
* [ImageNet](https://www.image-net.org/)
* ImageNet-21K Pretraining for the Masses
  
*** 

* 15:00

* **NOTE**:
  * We have self-attention
  * We have un-masked attention
    * No masking: BERT

***

* 25:00

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-11/vit-attention-block.png" width="500" height="300" />
</p>

* 12 Attention layers
* Dimention of each layer: $\frac{768}{12}=64$


#### Residual
* $Y=xW+\beta$
* Allow the input parameters to directly influence the output
* It means, Loss has a direct influence at the input (e.g., )
* If input tremendiously influences the predictor output then if there is difference between $output_{predicted}$ output and $output_{expected}$ that means there is a big loss.
* The Gradient of loss wrt to weights in the beingging $\frac{\partial L}{\partial \text{W}}$ will be a big number.
* It means, in backpropogation, the parameters at earlier part of NN will be changed considerably rather than it getting changed only insignificantly.
* If residual are not there then in a DNN ...

***

* 30:00

$$
\begin{align}
   Block ~A    & \rightarrow Block ~B \rightarrow Block ~C\\
   (3,224,224) & \rightarrow (196,788) \rightarrow\\
\end{align}
$$

* $(RGB,H,W)=(3,224,224)$

***

* 35:00

#### Code: ViT on MNIST
* SigLIP

***

* 50:00

| Scheme | Paper |
|---|---|
| CLIP | [Learning Transferable Visual Models From Natural Language Supervision - (Feb 201)](https://arxiv.org/abs/2103.00020) |
| SigLIP | [Sigmoid Loss for Language Image Pre-Training - (Sep 2023)](https://arxiv.org/abs/2303.15343) |
* **CLIP (Contrastive Language-Image Pre-training)** is a NN developed by OpenAI that bridges the gap between CV and NLP
* **SigLIP (Sigmoid Loss for Language-Image Pre-Training)** is an advanced vision-language model developed by Google that maps images and text into a shared embedding space for tasks like cross-modal retrieval and object search. It is a highly efficient, more scalable alternative to the standard CLIP model.

***

* 1:00:00

#### Whre Cross-Attentin Lives in Practice


* 1:20:00

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-11/clip-algorithm.png" width="500" height="300" />
</p>

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-11/vlm-family.png" width="500" height="300" />
</p>

***

* 1:45:00

#### Image Generation


* [Inception - A new frontier in LLM speed](https://www.inceptionlabs.ai/)

***

#### Diffusion

+ $x_0$: Original Image
+ $\epsilon$: Raondom Noide
+ $\bar{a_t}$ shrinks from $1 \rightarrow 0$ as t grows large.
+ $x_t$: Noisy image at time t.
+ $\epsilon$: Actual Noise
+ $\epsilon_{\theta}$: Predicted Noise by NN
  
||||
|---|---|---|
| Forward (Noising)     | $$x_t= \sqrt{\bar{a_t}}.x_0 + \sqrt{1-\bar{a_t}}.\epsilon$$ | | 
| Training (MSE Loss)              | $$\mathcal{L} = \parallel\epsilon - \epsilon_{\theta}(x_t,t)\parallel^2$$ | |
| Sampling (Generation) | $$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\bigg(x_t-\frac{\beta_t}{1-\bar{\alpha_t}}\epsilon_{\theta}(x_t,t)\bigg)+\sigma_tz$$ | |

***

* 2:05:00

* VAE variational autoencoder

* 2:15:00

***




