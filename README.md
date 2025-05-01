# 🧠 Brain MRI Classification via SimCLR & Cosine MLP

This project explores **contrastive pretraining and feature-based transfer learning** for brain MRI classification. By leveraging **SimCLR** for unsupervised feature extraction, we enable **flexible and efficient downstream classification** across multiple neurological conditions including **tumors and strokes**.

---


## Contributors
-  [`Anh Bui`](https://github.com/anhbui229)
-  [`Chin Vergara`](https://github.com/ChinMV)
-  [`Farouk Braham`](https://github.com/faroukb11)


---

## Why SimCLR?

Instead of training a model from scratch or fine-tuning a heavyweight transformer like **Facebook's DeiT**, we use **SimCLR** to learn general-purpose brain scan features from unlabeled data.

- **Once trained**, the SimCLR encoder acts as a **universal feature extractor**
- These embeddings can then be used for **any downstream task**:
  - Tumor detection
  - Stroke diagnosis
 
- ⚡ This approach is **parameter-efficient** and **much faster to train** than end-to-end supervised models

---


## 🧬 Datasets Used

We merged several publicly available brain MRI datasets:

- 🧠 [`youngp5/BrainMRI`](https://huggingface.co/datasets/youngp5/BrainMRI)
- 🧠 [`Mahadih534/brain-tumor-MRI-dataset`](https://huggingface.co/datasets/Mahadih534/brain-tumor-MRI-dataset)
- 🧠 [`Falah/Alzheimer_MRI`](https://huggingface.co/datasets/Falah/Alzheimer_MRI) (train/test)
- 🧠 [`BTX24/tekno21-brain-stroke-dataset-binary`](https://huggingface.co/datasets/BTX24/tekno21-brain-stroke-dataset-binary)

All images were resized, normalized, and converted to grayscale, then augmented for SimCLR pretraining.

---

## Framework versions

- **python**: 3.11.12
- **torch**: 2.6.0+cpu
- **torchvision**: 0.21.0+cpu
- **pytorch_lightning**: 2.5.1.post0
- **datasets**: 3.5.1

---

## Data Augmentations

We applied the following augmentations during SimCLR training to learn robust, structure-aware embeddings:

- `RandomResizedCrop`
- `GaussianBlur`
- `Grayscale`
- `RandomHorizontalFlip`
- `RandomAffine`
- `Normalize(mean=0.5, std=0.5)`

---

## Cosine MLP Classifier

After pretraining, we froze the SimCLR encoder and trained a **Cosine MLP Classifier** on top of the 512-dim features.
Instead of a standard linear head, we use a Cosine MLP Classifier, which computes classification scores based on cosine similarity between normalized embeddings and class weights.

---

## Architecture & Hyperparameters Summary

<table>
<tr>
<th style="text-align:center"> SimCLR Backbone</th>
<th style="text-align:center"> Cosine MLP Classifier</th>
</tr>
<tr>
<td>

###  Key Design Points

- Backbone: `torchvision.models.resnet18`
- Modified projection head:
  - `Linear(512 → 4×hidden_dim) → ReLU → Linear(4×hidden_dim → hidden_dim)`
- Contrastive learning using **InfoNCE loss**
- Uses **cosine similarity** for positive/negative comparison
- Learns representations from **unlabeled data**

###  Hyperparameters

- `hidden_dim = 128`
- `temperature = 0.07`
- `lr = 5e-4`
- `weight_decay = 1e-4`
- `batch_size = 512`
- `max_epochs = 200`
- `optimizer = AdamW`
- `scheduler = CosineAnnealingLR`

</td>
<td>

###  Key Design Points

- Lightweight head trained on frozen SimCLR features (512-dim)
- MLP encoder with batch norm and dropout:
  - `Linear(512 → 128) → BatchNorm → RELU → Dropout`
- Cosine-based classifier head:
  - `Linear(128 → num_classes, bias=False)`
  - Followed by cosine similarity + learnable scaling factor

###  Hyperparameters

- `hidden_dims = [512, 128]`
- `dropout = 0.3`
- `lr = 5e-4`
- `weight_decay = 1e-3`
- `batch_size = 256`
- `max_epochs = 150`
- `optimizer = AdamW`
- `scheduler = MultiStepLR (at 60% and 80% of training)`

</td>
</tr>
</table>

---

## Results
Despite training only the lightweight MLP head, we achieved strong performance across all tasks — matching or outperforming deeper fine-tuned models like DeiT with significantly fewer parameters and compute.

### Stroke Classification
<table> <tr> <td><strong>Confusion Matrix</strong></td> <td><strong>Classification Report</strong></td> </tr> <tr> <td><img src="images/stroke2.png" width="400"></td> <td><img src="images/stroke_cm.png" width="400"></td> </tr> </table>


### Tumor Detection
<table> <tr> <td><strong>Confusion Matrix</strong></td> <td><strong>Classification Report</strong></td> </tr> <tr> <td><img src="images/tumor.png" width="400"></td> <td><img src="images/tumor_cm.png" width="400"></td> </tr> </table>

---

##  References

This project builds on foundational and recent work in contrastive learning and medical image analysis. Key references include:

- **Chen, Ting, et al.** (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* International Conference on Machine Learning (ICML). [SimCLR Paper](https://arxiv.org/abs/2002.05709)

- **Chaitanya, Krishna, et al.** (2020). *Contrastive Learning of Global and Local Features for Medical Image Segmentation with Limited Annotations.* Advances in Neural Information Processing Systems (NeurIPS). [Link](https://arxiv.org/abs/2006.10511)

- **Azizi, Sohil, et al.** (2021). *Big Self-Supervised Models Advance Medical Image Classification.* International Conference on Computer Vision (ICCV). [Link](https://arxiv.org/abs/2101.05224)

- **Qin, Chen, et al.** (2022). *Self-Supervised Learning for Ischemic Stroke Lesion Segmentation from CT Perfusion Images.* Medical Image Analysis. [Link](https://doi.org/10.1016/j.media.2021.102247)

- **Wang, Zihan, et al.** (2022). *Generalizable and Data-Efficient Stroke Detection with Self-Supervised Learning.* Frontiers in Neurology. [Link](https://www.frontiersin.org/articles/10.3389/fneur.2022.905605)