#!/usr/bin/env python 

from typing import Union, List, Tuple
import logging
import os
import random

import pandas as pd
import torch
from tqdm import tqdm
from rdkit.Chem import AllChem
import torch_geometric.data as gdata

from chem_utils.chem_handler import RxnHandler
from rxn_template_selector.filter_tids_data import FilterTidsData
from data_utils.rxn_and_rxn_template_data import RxnAndRxnTemplateData
from utils.gnn_data_utils import GNNDataUtils
from config.config import Config
from config.rcts_config import RCTSConfig


class RCTDataset(gdata.InMemoryDataset):
    
    def __init__(self, train_or_test: str, rct_config: RCTSConfig):
        self._config = rct_config

        if train_or_test == "train":
            self._rids_and_tids_fp = self._config.train_rids_fp
            root = rct_config.train_temp_dp
        elif train_or_test == "test":
            self._rids_and_tids_fp = self._config.test_rids_fp
            root = rct_config.test_temp_dp
        else:
            raise AttributeError(f"Except 'train' or 'test', but get '{train_or_test}'")

        super(RCTDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['products.centralized_templates']

    def download(self):
        pass

    def process(self):
        if not os.path.exists(self._config.train_rids_fp):
            self._split_train_test_data()
        rxn_and_template_data = RxnAndRxnTemplateData(self._config.rxn_data_tsv_fp,
                                                      self._config.rct_tsv_fp,
                                                      self._rids_and_tids_fp,
                                                      "centralized")
        filter_tids_data = FilterTidsData(self._config.filter_tids_fp,
                                          self._config.min_num_covered_rxns_by_rct,
                                          self._config.rct_tsv_fp)
        data_list = []
        rids, tids = rxn_and_template_data.get_rids_and_tids_by_covered(self._config.min_num_covered_rxns_by_rct)
        with tqdm(total=len(rids))as pbar:
            for i, (rid, tid) in enumerate(zip(rids, tids)):
                rxn_smiles = rxn_and_template_data.get_rxn_smiles_by_rid(rid)

                rxn = AllChem.ReactionFromSmarts(rxn_smiles)
                rxn = RxnHandler.remove_unmapped_mols_in_rxn(rxn)
                rxn = RxnHandler.remove_products_same_with_reactants(rxn)
                if rxn.GetNumProductTemplates() != 1:
                    logging.warning(f"\nwrong reaction smiles {AllChem.ReactionToSmiles(rxn)} \nfor rid({rid}) and tid({tid})")
                    continue
                data = GNNDataUtils.get_gnn_data_from_smiles(AllChem.MolToSmiles(rxn.GetProducts()[0]))
                data.y = torch.tensor([filter_tids_data.get_one_hot_idx_by_tid(tid)], dtype=torch.long)
                data_list.append(data)

                pbar.update(1)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _split_train_test_data(self):
        logging.info(f"Splitting train and test to {self._config.train_rids_fp} and {self._config.test_rids_fp}")
        rid_and_tid_df = pd.read_csv(self._config.rid_tid_tsv_fp, sep='\t', encoding='utf-8')
        filter_rid_and_tid_df = rid_and_tid_df.query(f"c_num>={self._config.min_num_covered_rxns_by_rct}")
        filter_rid_and_tid_df = filter_rid_and_tid_df.sample(frac=1)
        train_num = int(len(filter_rid_and_tid_df) * 0.9)
        filter_rid_and_tid_df.iloc[:train_num].to_csv(self._config.train_rids_fp, sep='\t', encoding='utf-8', index=False)
        filter_rid_and_tid_df.iloc[train_num:].to_csv(self._config.test_rids_fp, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    config = Config("../config.json")
    RCTDataset("train", config.rcts_config)
    RCTDataset("test", config.rcts_config)
