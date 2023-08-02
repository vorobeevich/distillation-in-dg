from src.datasets import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VLCS(Dataset):
    def __init__(
            self,
            dataset_type: list[str],
            domain_list: list[str],
            **kwargs) -> None:
        super().__init__("vlcs", dataset_type, domain_list, **kwargs)