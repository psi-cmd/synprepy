#!/usr/bin/env python
import sys
sys.path.append('/root/synprepy/')

from gat_model import GATModel
from torch_geometric.data import DataLoader, Batch
from rxn_template_selector.filter_tids_data import FilterTidsData
from rxn_template_selector.rct_dataset import RCTDataset
from config.config import Config
import torch.nn.functional as F
from utils.model_utils import ModelUtils
from torch.optim import lr_scheduler
import torch
from tqdm import tqdm
import logging

from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)


class RCTTrainer:
    def __init__(self):
        config = Config("/root/synprepy/config.json")
        rcts_config = config.rcts_config
        batch_size = rcts_config.batch_size
        self._epoch_num = rcts_config.epoch_num
        lr_start = rcts_config.lr_start
        lr_end = rcts_config.lr_end
        self._device = rcts_config.device
        self._model_dp = rcts_config.model_dp

        _train_dataset = RCTDataset("train", rcts_config)
        self._train_dataloader = DataLoader(
            _train_dataset, batch_size=batch_size)

        _test_dataset = RCTDataset("test", rcts_config)
        self._test_dataloader = DataLoader(
            _test_dataset, batch_size=batch_size)

        self._filter_tids_data = FilterTidsData(rcts_config.filter_tids_fp)

        self._model = GATModel(
            self._filter_tids_data.num_tids).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=lr_start)
        self._scheduler = lr_scheduler.StepLR(
            self._optimizer, step_size=10, gamma=0.5, )

    def train(self):
        for epoch in range(self._epoch_num):
            train_loss, train_acc = self._train_epoch(epoch)
            test_loss, test_acc = self.test(epoch)
            self._scheduler.step()
            tb.add_scalar("train/loss", train_loss, epoch)
            tb.add_scalar("train/acc", train_acc, epoch)
            tb.add_scalar("test/loss", test_loss, epoch)
            tb.add_scalar("test/acc", test_acc, epoch)
            logging.info(f"\ntrain loss: {train_loss}, acc: {train_acc}\ntest loss: {test_loss}, acc: {test_acc}")
            ModelUtils.save_model(self._model_dp, self._model, epoch,
                                  train_loss, train_acc,
                                  test_loss, test_acc)

    def _train_epoch(self, epoch: int):
        self._model.train()
        tot_loss = 0
        tot_correct = 0
        tot_mol = 0
        with tqdm(total=len(self._train_dataloader))as pbar:
            pbar.set_description_str(f"train epoch: {epoch}")
            for n, batch_data in enumerate(self._train_dataloader):
                batch_data = batch_data.to(self._device)
                pred = self._model(batch_data)
                loss = F.cross_entropy(pred, batch_data.y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                num_mols = batch_data.num_graphs
                correct = pred.max(dim=1)[1].eq(batch_data.y).sum().item()
                tot_correct += correct
                tot_mol += num_mols
                tot_loss += (loss.item()*num_mols)
                pbar.set_postfix_str(
                    f"loss: {tot_loss / tot_mol}, acc: {tot_correct / tot_mol}")
                pbar.update(1)
        return tot_loss / tot_mol, tot_correct / tot_mol

    def test(self, epoch: int):
        self._model.eval()
        tot_loss = 0
        tot_correct = 0
        tot_mol = 0
        with torch.no_grad():
            with tqdm(total=len(self._test_dataloader))as pbar:
                pbar.set_description_str(f"test epoch: {epoch}")
                for n, batch_data in enumerate(self._test_dataloader):
                    batch_data = batch_data.to(self._device)
                    pred = self._model(batch_data)
                    loss = F.cross_entropy(pred, batch_data.y)

                    num_mols = batch_data.num_graphs
                    correct = pred.max(dim=1)[1].eq(batch_data.y).sum().item()
                    tot_correct += correct
                    tot_mol += num_mols
                    tot_loss += (loss.item() * num_mols)
                    pbar.set_postfix_str(
                        f"loss: {tot_loss / tot_mol}, acc: {tot_correct / tot_mol}")
                    pbar.update(1)
        return tot_loss / tot_mol, tot_correct / tot_mol

from utils.train_utils import sigint_ignored
import time

if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    tb = SummaryWriter(log_dir="/root/synprepy/logs/rct_GAT_train_" + now)
    with sigint_ignored():
        trainer = RCTTrainer()
        trainer._model.train()
        # tb.add_graph(trainer._model)
        trainer.train()
