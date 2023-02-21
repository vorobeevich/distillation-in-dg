import wandb

class WandbLogger:
    def __init__(self, config):
        self.config = config
        wandb.login()
        self.run = wandb.init(
            entity="distillation-generalization",
            project=config["wandb_project"],
            config=config,
        )
        wandb.run.name  = config["trainer"]["run_id"]
        

    def log_epoch(self, test_domain: str, figure_type: str, figure_name: str, value: float, num_step: int, commit: bool = False):
        self.run.log({f"{test_domain}.{figure_type}.{figure_name}":  value}, commit=commit)
