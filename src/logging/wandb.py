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

    def log_metric(self, test_domain: str, figure_type: str, figure_name: str, value: float, num_step: int):
        name = f"{test_domain}.{figure_type}.{figure_name}"
        self.run.log({name:  value, f"{name}.epoch" : num_step})

    def log_table(self, df):
        self.run.log({"Results": wandb.Table(daatframe=df)})
