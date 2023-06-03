import yaml
import random

import sys
sys.path.append("./")

from src.utils import fix_seed
fix_seed(42)

with open("src/configs/swad/distillation_large_teacher.yaml", "r") as stream:
    config = yaml.safe_load(stream)

n_configs = 20
temperatures = [1, 2, 5, 10]
learning_rates = [5e-4, 1e-5, 3e-5, 5e-5]
weight_decays = [1e-4, 1e-5, 1e-6]
all_combinations = []

for temperature in temperatures:
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            all_combinations.append((temperature, learning_rate, weight_decay))    
combinations = random.sample(all_combinations, n_configs)
random.shuffle(combinations)

for i in range(n_configs):
    temperature, learning_rate, weight_decay = combinations[i]
    config["temperature"] = temperature
    config["optimizer"]["kwargs"]["lr"] = learning_rate
    config["optimizer"]["kwargs"]["weight_decay"] = weight_decay
    with open(f"src/configs/swad/distillation_large_teacher_{i}.yaml", "w") as stream:
        yaml.dump(config, stream, default_flow_style=None)