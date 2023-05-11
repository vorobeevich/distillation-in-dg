import torch.utils.data

def num_iters_loader(loader: torch.utils.data.DataLoader, num_iters: int):
    iter = 0
    while iter < num_iters:
        for batch in loader:
            yield batch 
            iter += 1
            if iter == num_iters:
                break