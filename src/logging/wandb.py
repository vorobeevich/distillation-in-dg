import wandb
import pandas as pd


class WandbLogger:
    def __init__(self, config):
        self.config = config
        wandb.login()
        self.run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            config=config,
            name=config["run_id"]
        )

    def log_metric(self, name: str, value: float, num_step: int):
        self.run.log({name: value, f"{name}.epoch": num_step})

    def log_table(self, name: str, df: pd.DataFrame):
        self.run.log({name: wandb.Table(dataframe=df)})

    def log_image(self, image_array, name: str):
        self.run.log({name: wandb.Image(image_array)})
    
    def watch(self, model, loss_function, log, log_freq):
        self.run.watch(model, loss_function, log, log_freq)
