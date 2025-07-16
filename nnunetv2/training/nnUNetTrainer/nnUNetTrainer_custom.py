import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0

# Define a custom trainer class for nnUNet
class nnUNetTrainer_custom(nnUNetTrainer_DASegOrd0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.save_every = 1
        self.oversample_foreground_percent = 0.5
        # 247738 training and 5965 validation cases.
        
        self.num_iterations_per_epoch = 1000  # Number of iterations per epoch
        # number of batches per epoch = batch_size * num_iterations_per_epoch
        self.num_val_iterations_per_epoch = 50
        
        self.initial_lr = 1e-3  # Initial learning rate
        self.weight_decay = 5e-4  # Weight decay for regularization
        
        
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                   T_0=10,
                                                   T_mult=1,
                                                   eta_min=1e-6,
                                                   last_epoch=-1)
        return optimizer, lr_scheduler