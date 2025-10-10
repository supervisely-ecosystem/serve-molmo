<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/serve-molmo/releases/download/v0.0.1/molmo.png"/>  

# Serve Molmo

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../supervisely-ecosystem/serve-molmo)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-molmo)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/serve-molmo.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/serve-molmo.png)](https://supervisely.com)

</div>

# Overview

Molmo (Multimodal Open Language Model) is afamily of state-of-the-art open VLMs. Molmo architecture follows a standard design, combining pre-trained language and vision models. It has four components: (1) a pre-processor that converts the input image into multiscale, multi-crop images, (2) a ViT image encoder that computes per-patch features for each image independently, (3) a connector that pools and projects patch features into the LLM’s embedding space, and (4) a decoder-only LLM.

![Molmo](https://github.com/supervisely-ecosystem/serve-molmo/releases/download/v0.0.1/molmo_architecture.png)

Most ViTs only support square images at a fixed resolution that is generally too low for fine-grained tasks such as OCR or detailed captioning. To address this issue input images are divided into multiple square crops that tile the image. Additionally, the full image, resized to the ViT’s resolution, provides a low-resolution overview. Each crop is processed independently by the ViT.

Once crops are encoded by the vision encoder, patch features are built by concatenating features from the third-to-last and tenth-from-last ViT layers, which improves performance slightly over using a single layer. Each 2×2 patch window is then pooled into a single vector using multi-headed attention layer, where the mean of the patches serves as the query. This attention pooling outperforms simple feature concatenation. Finally, pooled features are mapped to the LLM’s embedding space via an MLP.

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/serve-molmo/releases/download/v0.0.1/deploy_molmo.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/serve-molmo/releases/download/v0.0.1/deploy_molmo_2.png)

# Acknowledgment

This app is based on the great work [Molmo](https://github.com/allenai/molmo). ![GitHub Org's stars](https://img.shields.io/github/stars/allenai/molmo?style=social)
