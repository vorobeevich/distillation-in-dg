# Introduction
Official implementation of **[Weight Averaging Improves Knowledge Distillation under Domain Shift](https://arxiv.org/abs/2309.11446)**

Valeriy Berezovskiy, Nikita Morozov

[ICCV 2023 Out Of Distribution Generalization in Computer Vision Workshop](https://www.ood-cv.org/)

# Preparation

We **highly** recommend using [_conda_](https://www.anaconda.com/download) for experiments.

After installation, make a new environment:

```conda create python=3.10 --name dist --yes```

```conda activate dist```

Install libs from _requirements.txt_:

```conda install --file requirements.txt --yes```

[_Torch_](<https://pytorch.org/get-started/locally/>) versions may differ depending on your **GPU**.

# Data

Load **PACS** and **Office-Home** datasets:

```chmod 777 ./src/scripts/load_pacs.sh ./src/scripts/load_officehome.sh```

```./src/scripts/load_pacs.sh && ./src/scripts/load_officehome.sh```

# Usage

**independent** learning on Cross Entropy:

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[SELECT CONFIG TO RUN] --test_domain [TEST_DOMAINS SETS]```

**distillation:**

```python src/scripts/train.py --device [ID OF CUDA DEVICE] --config src/configs/[SELECT CONFIG TO RUN] --test_domain [TEST_DOMAINS SETS] --dist```

For **PACS** _art_painting, photo, sketch, cartoon_ domains are available to select. You can select several at once: _--test photo cartoon_. For **Office-Home** _art, clipart, product, real_world_ domains are available to select. 
Check id of required **GPU** device using ```nvidia-smi``` command. Before starting distillation, you need to train config with the **teacher** model. 

Let's look at the config [structure](https://github.com/vorobeevich/distillation-in-dg/blob/main/src/configs/pacs/swad/student_baseline_1.yaml). At the end of the config name there is a random seed. You can use any model or augmentation from _torchvision_. For the model, it is necessary to include parameters of the last linear layer. Also [_DeiT_](https://huggingface.co/docs/transformers/model_doc/deit) model is avaliable. For the dataset, you must specify name (**PACS**, **OfficeHome**) and list of domains. Also, you can change training parameters: any optimizer from _torch.optim_, batch size, **SWAD** parameters, and so on.

# Visualization

All experiments were logged using [_wandb_](<https://wandb.ai/gegelyanec/dist-gen?workspace=user-gegelyanec>) library: .

# Citation

```
@article{berezovskiy2023weight,
  title={Weight Averaging Improves Knowledge Distillation under Domain Shift},
  author={Berezovskiy, Valeriy and Morozov, Nikita},
  journal={arXiv preprint arXiv:2309.11446},
  year={2023}
}
```

# Contact

If you have any questions, feel free to contact us through email (vsberezovsksiy@edu.hse.ru or nmorozov@hse.ru).
