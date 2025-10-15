from posteriors.sgld_lightning import SGLDClassification
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.viz_utils import (
    plot_training_metrics,
)
import tempfile
from lightning import LightningDataModule
import os

import torch
from torch.utils.data import DataLoader


class sgld(object):
    def __init__(self, net, loss_fn, lr, wd, nf, epochs, res_dir=None):
        self.net = net
        self.device = next(self.net.parameters()).device
        self.loss_fn = loss_fn
        self.lr = lr
        self.wd = wd
        self.nf = nf
        self.epochs = epochs
        self.sgldmodel = SGLDClassification(model=self.net,
                                    loss_fn=self.loss_fn,
                                    lr=self.lr,
                                    weight_decay=self.wd,
                                    noise_factor=self.nf)
        
        self.my_temp_dir = tempfile.mkdtemp(res_dir)

        self.logger = CSVLogger(self.my_temp_dir)

        self.trainer = Trainer(
            max_epochs=self.epochs,  # number of epochs we want to train
            logger=self.logger,  # log training metrics for later evaluation
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=True,
            default_root_dir=self.my_temp_dir,
        )
    
    def train(self, train_data, test_data, val_data, batch_size, plot_loss=False):
        lightningdata = LightningDataset(train=train_data,
                                        test=test_data,
                                        val=val_data,
                                        train_bs=batch_size,
                                        test_bs=batch_size,
                                        val_bs=batch_size
                    )
        
        self.trainer.fit(self.sgldmodel, lightningdata)

        if plot_loss:
            fig = plot_training_metrics(
                os.path.join(self.my_temp_dir, "lightning_logs"), ["train_loss", "trainAcc"]
            )
            fig.savefig(os.path.join(self.my_temp_dir, "lightning_logs.pdf"), format='pdf')
        
    def eval(self, x):
        return self.sgldmodel.predict_step(x.to(self.device), device=self.device)['logits']


def collate_fn_tensordataset(batch):
    """Collate function for tensor dataset to our framework."""
    inputs = torch.stack([item[0] for item in batch])
    targets = torch.stack([torch.tensor(item[1]) for item in batch])
    return {"input": inputs, "target": targets}

class LightningDataset(LightningDataModule):

    def __init__(
        self,
        train,
        test,
        val,
        train_bs,
        test_bs,
        val_bs
    ) -> None:
        """Initialize a new Toy Data Module instance.
        """
        super().__init__()

        self.train = train
        self.test = test
        self.val = val
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.val_bs = val_bs

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(self.train,self.train_bs, collate_fn=collate_fn_tensordataset)

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader."""
        return DataLoader(self.test,self.test_bs, collate_fn=collate_fn_tensordataset)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(self.val,self.val_bs, collate_fn=collate_fn_tensordataset)