import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import TrainDataset
from net.MEASNet import IRmodel
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class IRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = IRmodel()
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, loss_b = self.net(degrad_patch)
        loss = self.loss_fn(restored,clean_patch) + 0.001*loss_b
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)
        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")
    trainset = TrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset,
                             batch_size=opt.batch_size,
                             pin_memory=True,
                             shuffle=True,
                             drop_last=True,
                             num_workers=opt.num_workers)
    model = IRModel()
    trainer = pl.Trainer( max_epochs=opt.epochs,
                          accelerator="gpu",
                          devices=opt.num_gpus,
                          strategy="ddp_find_unused_parameters_true",
                          logger=logger,
                          callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    import os
    os.environ["WANDB_DISABLED"] = "true"
    main()