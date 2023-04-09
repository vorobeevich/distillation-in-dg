import yaml
import argparse
import typing as tp

from src.parser.parser import Parser


class DistillParser(Parser):
    """Static class (a set of methods for which inheritance is possible) for parsing command line arguments
    and model training configuration.
    """

    @staticmethod
    def parse_config(args: argparse.Namespace) -> dict[str, tp.Any]:
        trainer_params = Parser.parse_config(args)
        # read config_yaml from path
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
        trainer_params["model_teacher_config"] = config["model_teacher"]
        for param in ["temperature", "run_id_teacher", "mixup"]:
            trainer_params[param] = config[param]

        return trainer_params
