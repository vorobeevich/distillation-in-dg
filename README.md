# distillation-generalization
The source code of "Empirical Study of Knowledge Distillation in Domain Generalization Tasks" paper.

# Usage
We **highly** recommend using conda for experiments: <https://www.anaconda.com/download>.

After installation, make a new environment:

```conda create --name dist```

```conda activate dist```

Install the libraries from the requirements.txt. Torch versions may differ depending on your GPU: <https://pytorch.org/get-started/locally/>



**independent** learning on Cross Entropy:

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[CHOOSE CONFIG TO RUN]```

**distillation:**

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[CHOOSE CONFIG TO RUN] --dist```

# Visualization

All experiments were logged using wandb library: <https://wandb.ai/gegelyanec/dist-gen?workspace=user-gegelyanec>.
 
