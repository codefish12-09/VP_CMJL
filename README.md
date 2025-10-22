The code is being organized. We will make it available as soon as possible.
# [ICCV 2025] Learning Visual Proxy for Compositional Zero-Shot Learning
* **Title**: **[Learning Visual Proxy for Compositional Zero-Shot Learning](https://arxiv.org/pdf/2501.13859)**
* **Authors**: Shiyu Zhang, Cheng Yan1, Yang Liu, Chenchen Jing, Lei Zhou, Wenjun Wang
* **Institutes**: Tianjin University, Zhejiang University, Zhejiang University of Technology, Hainan University
## Overview
![](https://github.com/codefish12-09/VP_CMJL/blob/main/images/method.jpg?raw=true)
## Description
Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions by leveraging
knowledge from seen compositions. Current methods align textual prototypes with visual features via Vision-Language Models (VLMs), but suffer from two limitations: 
(1) Modality gaps hinder the discrimination of semantically similar pairs.
(2) Single-modal textual prototypes lack fine-grained visual cues. 
In this paper, we introduce Visual Proxy Learning, a method that reduces modality gaps and enhances compositional generalization. We initialize visual proxies for attributes, objects, and their compositions using text representations and optimize the visual space to capture
fine-grained cues, improving visual representations. Additionally, we propose Cross-Modal Joint Learning (CMJL), which imposes cross-modal constraints between the textimage and fine-grained visual spaces, improving generalization for unseen compositions and discriminating similar pairs. Experiments show state-of-the-art performance in closed-world scenarios and competitive results in open-world settings across four CZSL benchmarks, demonstrating the effectiveness of our approach in compositional generalization.
## Results

### Main Results

The following results are obtained with a pre-trained CLIP (ViT-L/14). More experimental results can be found in the paper.
![](https://github.com/codefish12-09/VP_CMJL/blob/main/images/experiment.png?raw=true)
