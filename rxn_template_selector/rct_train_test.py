#!/usr/bin/env python 
import logging

from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader, Batch

from utils.torch_utils import TorchUtils
from config.config import Config
from utils.model_utils import ModelUtils
from rxn_template_selector.rct_dataset import RCTDataset
from rxn_template_selector.filter_tids_data import FilterTidsData
from rxn_template_selector.gnn_model import GNNModel
from utils.topk_acc_calculator import TopkAccCalculator

logging.basicConfig(level=logging.INFO)


class RCTTrainTest:

    def __init__(self):
        config = Config("../config.json")
        rcts_config = config.rcts_config
        batch_size = rcts_config.batch_size
        self._epoch_num = rcts_config.epoch_num
        lr_start = rcts_config.lr_start
        lr_end = rcts_config.lr_end
        self._device = rcts_config.device
        self._model_dp = rcts_config.model_dp

        _train_dataset = RCTDataset("train", rcts_config)
        self._train_dataloader = DataLoader(_train_dataset, batch_size=batch_size)

        _test_dataset = RCTDataset("test", rcts_config)
        self._test_dataloader = DataLoader(_test_dataset, batch_size=batch_size)

        self._filter_tids_data = FilterTidsData(rcts_config.filter_tids_fp)

        self._model = GNNModel(_train_dataset.num_node_features,
                               _train_dataset.num_edge_features,
                               self._filter_tids_data.num_tids).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr_start)
        self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size=10, gamma=0.5, )
        # self._scheduler = lr_scheduler.ReduceLROnPlateau(self._optimizer, 'min', factor=0.5, min_lr=lr_end)

    def train(self):
        for epoch in range(self._epoch_num):
            train_loss, train_acc = self._train_epoch(epoch)
            test_loss, test_acc = self.test(epoch)
            self._scheduler.step()
            logging.info(f"\ntrain loss: {train_loss}, acc: {train_acc}\n"
                         f"test loss: {test_loss}, acc: {test_acc}")
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
                g, pred = self._model(batch_data)
                loss = F.cross_entropy(pred, batch_data.y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                num_mols = batch_data.num_graphs
                correct = pred.max(dim=1)[1].eq(batch_data.y).sum().item()
                tot_correct += correct
                tot_mol += num_mols
                tot_loss += (loss.item()*num_mols)
                pbar.set_postfix_str(f"loss: {tot_loss / tot_mol}, acc: {tot_correct / tot_mol}")
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
                    g, pred = self._model(batch_data)
                    loss = F.cross_entropy(pred, batch_data.y)

                    num_mols = batch_data.num_graphs
                    correct = pred.max(dim=1)[1].eq(batch_data.y).sum().item()
                    tot_correct += correct
                    tot_mol += num_mols
                    tot_loss += (loss.item() * num_mols)
                    pbar.set_postfix_str(f"loss: {tot_loss / tot_mol}, acc: {tot_correct / tot_mol}")
                    pbar.update(1)
        return tot_loss / tot_mol, tot_correct / tot_mol

    def eval(self):
        self._model.load_state_dict(ModelUtils.load_best_model(self._model_dp))
        tac = TopkAccCalculator(60)
        self._model.eval()
        with torch.no_grad():
            with tqdm(total=len(self._test_dataloader))as pbar:
                pbar.set_description_str(f"eval")
                for n, batch_data in enumerate(self._test_dataloader):
                    batch_data = batch_data.to(self._device)
                    g, pred = self._model(batch_data)
                    tac.extend_pred_and_real(pred, batch_data.y)
                    pbar.update(1)
        logging.info(f"\n"
                     f"top1: {tac.get_topk_acc(1)}\n"
                     f"top5: {tac.get_topk_acc(5)}\n"
                     f"top10: {tac.get_topk_acc(10)}\n"
                     f"top20: {tac.get_topk_acc(20)}\n"
                     f"top50: {tac.get_topk_acc(50)}")


if __name__ == "__main__":
    RCTTrainTest().train()
    # RCTTrainTest().eval()
