<div align="center">

# [MICCAI 2025] Pathology-aware Virtual H\&E Staining of Section-free Thick Tissues with Semantic Contrastive Guidance</div>

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
