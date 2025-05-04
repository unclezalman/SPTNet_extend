# SPTNet: An Efficient Alternative Framework for Generalized Category Discovery with Spatial Prompt Tuning (ICLR 2024)


<p align="center">
    <a href="https://arxiv.org/abs/2403.13684"><img src="https://img.shields.io/badge/arXiv-2403.13684-b31b1b"></a>
    <a href="https://visual-ai.github.io/sptnet/"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
    <a href="https://huggingface.co/whj363636/SPTNet"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=models&color=yellow"></a>
    <a href="#jump"><img src="https://img.shields.io/badge/Citation-8A2BE2"></a>
</p>
<p align="center">
	SPTNet: An Efficient Alternative Framework for Generalized Category Discovery with Spatial Prompt Tuning <br>
  By
  <a href="https://whj363636.github.io/">Hongjun Wang</a>, 
  <a href="https://sgvaze.github.io/">Sagar Vaze</a>, and 
  <a href="https://www.kaihan.org/">Kai Han</a>.
</p>

![teaser](assets/teaser.png)

## Extension

This is the extended version of SPTNet created for the APAI3010_STAT3010 group project, by group 9. The extensions include adding the ability to run the model on the Food-101 dataset, as well as a multimodal version of SPTNet that integrates fixed text prompts with visual prompts that can be run on the CIFAR-100 dataset. 

## Prerequisite 🛠️

First, you need to clone the SPTNet repository from GitHub. Open your terminal and run the following command:

```
git clone https://github.com/Visual-AI/SPTNet.git](https://github.com/unclezalman/SPTNet_extend.git
cd SPTNet_extend
```

We recommend setting up a conda environment for the project. You will also need to download a CLIP backbone if using the extended model:

```bash
conda create -n spt_extend
conda activate spt_extend
pip install -r requirements.txt
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install git+https://github.com/openai/CLIP.git
```

## Running 🏃
### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

Generic object recognition datasets, including CIFAR-10/100 and ImageNet-100/1K:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100/1K](https://image-net.org/download.php)

Fine-grained benchmarks (CUB, Stanford-cars, FGVC-aircraft, Herbarium-19, Food-101). You can find the datasets in:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6) and [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101)


### Scripts

**Train the model(original)**:

```
CUDA_VISIBLE_DEVICES=0 python train_spt.py \
    --dataset_name 'food-101' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 1000 \
    --num_workers 8 \
    --sup_weight 0.35 \
    --weight_decay 5e-4 \
    --transform 'imagenet' \
    --lr 0.5 \
    --lr2 1e-2 \
    --prompt_size 1 \
    --freq_rep_learn 20 \
    --pretrained_model_path ${YOUR_OWN_PRETRAINED_PATH} \
    --prompt_type 'all' \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 10 \
    --memax_weight 1 \
    --model_path ${YOUR_OWN_SAVE_DIR}
```

**Eval the model(original)**
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --dataset_name 'food-101' \
    --pretrained_model_path ./checkpoints/fgvc/dinoB16_best.pt \
    --prompt_type 'all' \ # switch to 'patch' for 'cifar10' and 'cifar100'
    --eval_funcs 'v2' \
```

**Train the model(extended)**:

```
CUDA_VISIBLE_DEVICES=0 python train_spt_extend.py \
    --dataset_name 'cifar100' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 1000 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-4 \
    --transform 'imagenet' \
    --lr 5.0 \
    --lr2 3e-3 \
    --prompt_size 1 \
    --freq_rep_learn 20 \
    --pretrained_model_path './pretrained_models/clip/ViT-B-16.pt' \
    --prompt_type 'patch' \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 10 \
    --memax_weight 1 \
    --model_path ${YOUR_OWN_SAVE_DIR}\
    --model 'clip'
```

**Eval the model(extended)**
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --dataset_name 'cifar-100' \
    --pretrained_model_path ./checkpoints/dinoB16_best.pt \
    --prompt_type 'patch' \ 
    --eval_funcs 'v2' \
```

To reproduce the results in the project for Food-101, the model will have to be trained from scratch. The backbone model used for the results given can be found [here](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) 

Please note that the extended model can only be used for CIFAR-100 but could be edited further to be used on the other datasets from the original paper. 

## Hyper-parameters
These are the results and hyperparameters used for the original SPTNet, with no text prompt extension:
|              | lr1  | lr2  |memax|
|--------------|------|------|------
| CIFAR-10     | 1    | 3e-3 | 1   |
| CIFAR-100    | 5    | 3e-3 | 1   |
| ImageNet-100 | 5    | 3e-3 | 1   |
| CUB          | 25   | 5e-2 | 2   |
| SCARS        | 10   | 5e-2 | 1   |
| Aircraft     | 1    | 5e-2 | 1   |
| Herbarium19  | 0.5  | 1e-2 | 1   |
| Food-101     | 0.5  | 1e-2 | 1   |


## Results
Generic results:
|              | All  | Old  | New  |
|--------------|------|------|------|
| CIFAR-10     | 97.3 | 95.0 | 98.6 |
| CIFAR-100    | 81.3 | 84.3 | 75.6 |
| ImageNet-100 | 85.4 | 93.2 | 81.4 |

Fine-grained results:
|               | All  | Old  | New  |
|---------------|------|------|------|
| CUB           | 65.8 | 68.8 | 65.1 |
| Stanford Cars | 59.0 | 79.2 | 49.3 |
| FGVC-Aircraft | 59.3 | 61.8 | 58.1 |
| Herbarium19   | 43.4 | 58.7 | 35.2 |

New dataset results: 
|               | All  | Old  | New  |
|---------------|------|------|------|
| Food-101      | 66.0 | 89.7 | 40.0 |

Please note that for the results for Food-101 shown here, the model was trained for 50 epochs, with 5 warmup teacher temp epochs. All other parameters were the same as in the example script given.

## Extension results
|               | All  | Old  | New  |
|---------------|------|------|------|
| CIFAR-100     | 74.0 | 82.6 | 56.7 |

Please note that for the results shown for the extended SPTNet here, the model was trained for 50 epochs, with 5 warmup teacher epochs. All other parameters were the same as in the example script given, using train_spt_extend.py 
