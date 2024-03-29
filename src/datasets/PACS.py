from src.datasets import Dataset


class PACS(Dataset):
    def __init__(
            self,
            dataset_type: list[str],
            domain_list: list[str],
            **kwargs) -> None:
        super().__init__("pacs", dataset_type, domain_list, **kwargs)