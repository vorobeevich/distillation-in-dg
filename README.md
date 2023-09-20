# Intro
This repository contains the implementation code for paper:

**Weight Averaging Improves Knowledge Distillation under Domain Shift**

Valeriy Berezovskiy, Nikita Morozov

# Data

Load the necessary datasets using the commands:

```chmod ./src/scripts/load_pacs.sh 777```
```./src/scripts/load_pacs.sh```

# Usage
We **highly** recommend using conda for experiments: <https://www.anaconda.com/download>.

After installation, make a new environment:

```conda create --name dist```

```conda activate dist```

Install the libraries from the requirements.txt. Torch versions may differ depending on your GPU: <https://pytorch.org/get-started/locally/>

**independent** learning on Cross Entropy:

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[CHOOSE CONFIG TO RUN] --test_domain [TEST_DOMAINS SETS]```

**distillation:**

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[CHOOSE CONFIG TO RUN] --test_domain [TEST_DOMAINS SETS] --dist```

# Visualization

All experiments were logged using wandb library: <https://wandb.ai/gegelyanec/dist-gen?workspace=user-gegelyanec>.
 
