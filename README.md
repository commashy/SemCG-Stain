<div align="center">

<!-- # <b></b> Pathology-aware Virtual H\&E Staining of Section-free Thick Tissues with Semantic Contrastive Guidance</div>-->
# Pathology-aware Virtual H\&E Staining of Section-free Thick Tissues with Semantic Contrastive Guidance</div>
<!--This is the official repository of "[Progressive Knowledge Distillation for Automatic Perfusion Parameter Maps Generation from Low Temporal Resolution CT Perfusion Images](https://link.springer.com/chapter/10.1007/978-3-031-72117-5_57)," presented at **MICCAI 2024**.
> **Progressive Knowledge Distillation for Automatic Perfusion Parameter Maps Generation from Low Temporal Resolution CT Perfusion Images** <br>
> [Moo Hyun (Kyle) Son](mailto:mhson@cse.ust.hk)<sup>1</sup>, [Juyoung (Justin) Bae](mailto:jbaeaa@cse.ust.hk)<sup>1</sup>, [Elizabeth Tong](mailto:etong@stanford.edu)<sup>2</sup>, [Hao Chen](https://www.cse.ust.hk/~haochen/)<sup>1</sup> <br>
> <sup>1</sup>The Hong Kong University of Science and Technology (HKUST)&nbsp;&nbsp;<sup>2</sup>Stanford University <br>
>
> **Abstract.** Perfusion Parameter Maps (PPMs), generated from Computer Tomography Perfusion (CTP) scans, deliver detailed measurements of cerebral blood flow and volume, crucial for the early identification and strategic treatment of cerebrovascular diseases. However, the acquisition of PPMs involves significant challenges. Firstly, the accuracy of these maps heavily relies on the manual selection of Arterial Input Function (AIF) information. Secondly, patients are subjected to considerable radiation exposure during the scanning process. In response, previous researches have attempted to automate AIF selection and reduce radiation exposure of CTP by lowering temporal resolution, utilizing deep learning to predict PPMs from automated AIF selection and temporal resolutions as low as $\frac{1}{3}$. However, the effectiveness of these approaches remains marginally significant. In this paper, we push the limits and propose a novel framework, Progressive Knowledge Distillation (PKD), to generate accurate PPMs from $\frac{1}{16}$ standard temporal resolution CTP scans. PKD uses a series of teacher networks, each trained on different temporal resolutions, for knowledge distillation. Initially, the student network learns from a teacher with low temporal resolution; as the student is trained, the teacher is scaled to a higher temporal resolution. This progressive approach aims to reduce the large initial knowledge gap between the teacher and the student. Experimental results demonstrate that PKD can generate PPMs comparable to full-resolution ground truth, outperforming current deep learning frameworks.

 [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72117-5_57) ·  [Code](https://github.com/mhson-kyle/progressive-kd) · [Poster.](assets/poster.pdf)
>-->
<p align="center"><img src = assets/overall.png width="85%" height="85%"></p>

## Getting Started

To get started with this project, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/commashy/SemCG-Stain.git
cd SemCG-Stain
```

### Requirements
Before Training the model, make sure you have the following requirements installed:

```bash
pip install -r requirements.txt
```
### Training
1. Prepare your dataset in the required format
2. Adjust the configuration files to suit your training needs
3. Run the following command to train the model:

```bash
python train.py \
  --dataroot /directorytotraindataset \
  --model semcg_stain \
  --netG semCG \
  --lr 0.00002 \
  --netD multi \
  --use_clip_contrast \
  --use_hard_neg \
  --name <name_of_trial>
```

### Testing
To evaluate the model, run the following command:

```bash
python test.py \
  --dataroot /directorytotestdataset \
  --model semcg_stain \
  --netG semCG \
  --name <name_of_trained_trial> \
  --epoch <epoch_to_test> \
  --num_test <number_of_testing_set>
```

### Datasets and Preprocessing
Please refer to the paper for more details on the datasets and preprocessing steps.
Will be updated soon.

### Model Weights
Will be updated soon.

## Results

Below shows virtual staining results using SemCG-Stain.
![Results of Progressive Knowledge Distillation](assets/results.png)

<!--
## Citation
If you find this repository useful, please consider citing:

``` bibtex

}
```
-->
