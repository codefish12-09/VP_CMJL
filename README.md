# [ICCV 2025] Learning Visual Proxy for Compositional Zero-Shot Learning
* **Title**: **[Learning Visual Proxy for Compositional Zero-Shot Learning](https://arxiv.org/pdf/2501.13859)**
* **Authors**: Shiyu Zhang, Cheng Yan, Yang Liu, Chenchen Jing, Lei Zhou, Wenjun Wang
* **Institutes**: Tianjin University, Zhejiang University, Zhejiang University of Technology, Hainan University
## 🚀 Overview
![](https://github.com/codefish12-09/VP_CMJL/blob/main/images/method.jpg?raw=true)
## 📖 Description
Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions by leveraging
knowledge from seen compositions. Current methods align textual prototypes with visual features via Vision-Language Models (VLMs), but suffer from two limitations: 
(1) Modality gaps hinder the discrimination of semantically similar pairs.
(2) Single-modal textual prototypes lack fine-grained visual cues. 
In this paper, we introduce Visual Proxy Learning, a method that reduces modality gaps and enhances compositional generalization. We initialize visual proxies for attributes, objects, and their compositions using text representations and optimize the visual space to capture
fine-grained cues, improving visual representations. Additionally, we propose Cross-Modal Joint Learning (CMJL), which imposes cross-modal constraints between the text-image and fine-grained visual spaces, improving generalization for unseen compositions and discriminating similar pairs. Experiments show state-of-the-art performance in closed-world scenarios and competitive results in open-world settings across four CZSL benchmarks, demonstrating the effectiveness of our approach in compositional generalization.
## 📈 Results

### Main Results

The following results are obtained with a pre-trained CLIP (ViT-L/14). More experimental results can be found in the paper.
![](https://github.com/codefish12-09/VP_CMJL/blob/main/images/experiment.png?raw=true)

## ⚙️ Setup

Our work is implemented in PyTorch framework. Create a conda environment `vpcmjl` using:

```
conda create --name vpcmjl python=3.10.3
conda activate vpcmjl
pip install -r requirements.txt
```
## 📚 Download Dataset
We evaluate our method on four benchmark datasets commonly used in Compositional Zero-Shot Learning (CZSL):

MIT-States

UT-Zappos

C-GQA

VAW-CZSL

Download Instructions
Please follow the data preparation protocols provided by  [Troika](https://github.com/bighuang624/Troika)(for MIT-States, UT-Zappos, and C-GQA) and [OADis](https://github.com/nirat1606/OADis) (for VAW-CZSL).

## 🏋️ Training Phase

```py
python train_multi_proxy.py --dataset <DATASET> 
```

## 📊 Evaluation

We evaluate our models in two settings: closed-world and open-world.

### Closed-World Evaluation

```py
python test_multi_proxy.py --dataset <DATASET> 
```

### Open-World Evaluation

For our open-world evaluation, we compute the feasbility calibration and then evaluate on the dataset.

For feasbility calibration, we have computed feasibility similarities and saved them at `data/feasibility_<dataset>.pt`. Therefore, you don't need to handle this yourself. If you need to compute on your own, please refer to [DFSP](https://github.com/Forest-art/DFSP?tab=readme-ov-file#feasibility-calibration).

Just run:

```py
python test_multi_proxy.py \
--dataset <DATASET>  \
--open_world True \
--threshold 0.4 \
```
## 🙏 Acknowledgement

Thanks for the publicly available code of [Troika](https://github.com/bighuang624/Troika), [DFSP](https://github.com/Forest-art/DFSP?tab=readme-ov-file#feasibility-calibration)
