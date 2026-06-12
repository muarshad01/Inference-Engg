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
* Key point: We need a tokenizer for pixels. ViT calls it the patch embedding.

| Paper |
|---|
| [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - (June 2021)](https://arxiv.org/abs/2010.11929) |

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-11/vit_patch_embedding.png" width="500" height="300" />
</p>

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

