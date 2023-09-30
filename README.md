# Introduction
Official implementation of **[Weight Averaging Improves Knowledge Distillation under Domain Shift](https://arxiv.org/abs/2309.11446)**

Valeriy Berezovskiy, Nikita Morozov

[ICCV 2023 Out Of Distribution Generalization in Computer Vision Workshop](https://www.ood-cv.org/)

# Preparation

We **highly** recommend using [conda](https://www.anaconda.com/download) for experiments.

After installation, make a new environment:

```conda create python=3.10 --name dist --yes```

```conda activate dist```

Install the libraries from the requirements.txt:

```conda install --file requirements.txt --yes```

[Torch](<https://pytorch.org/get-started/locally/>) versions may differ depending on your GPU.

# Data

Load PACS and Office-Home datasets using the commands:

```chmod 777 ./src/scripts/load_pacs.sh ./src/scripts/load_officehome.sh```

```./src/scripts/load_pacs.sh && ./src/scripts/load_officehome.sh```

# Usage

**independent** learning on Cross Entropy:

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[CHOOSE CONFIG TO RUN] --test_domain [TEST_DOMAINS SETS]```

**distillation:**

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[CHOOSE CONFIG TO RUN] --test_domain [TEST_DOMAINS SETS] --dist```

Check id of the required GPU device using ```nvidia-smi``` command. Before starting distillation, you need to train the config with the **teacher** model. 

Let's look at the config structure:

# Visualization

All experiments were logged using [wandb](<https://wandb.ai/gegelyanec/dist-gen?workspace=user-gegelyanec>) library: .

# Citation

```
@article{berezovskiy2023weight,
  title={Weight Averaging Improves Knowledge Distillation under Domain Shift},
  author={Berezovskiy, Valeriy and Morozov, Nikita},
  journal={arXiv preprint arXiv:2309.11446},
  year={2023}
}


```
 
