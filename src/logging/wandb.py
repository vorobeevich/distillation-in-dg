import wandb


class WandbLogger:
    def __init__(self, config):
        self.config = config
        wandb.login()
        self.run = wandb.init(
            project=config["wandb_project"],
            config=config,
        )
        wandb.run.name  = config["trainer"]["run_id"]
        

    def log_epoch(self, figure_name: str, value: float, num_epoch: int):
        self.run.log({figure_name: value, "epoch": num_epoch})