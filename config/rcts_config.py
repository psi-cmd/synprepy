#!/usr/bin/env python 

from config.config_base import ConfigBase


class RCTSConfig(ConfigBase):

    def __init__(self, config_json, rxn_data_tsv_fp: str, rct_tsv_fp: str, rid_tid_tsv_fp: str):
        self.rxn_data_tsv_fp = rxn_data_tsv_fp
        self.rct_tsv_fp = rct_tsv_fp
        self.rid_tid_tsv_fp = rid_tid_tsv_fp
        
        super(RCTSConfig, self).__init__(config_json)

        self.filter_tids_fp = self.get_file_path_by_key("filter_tids_file_name")
        self.train_rids_fp = self.get_file_path_by_key("train_rids_file_name")
        self.test_rids_fp = self.get_file_path_by_key("test_rids_file_name")
        self.train_temp_dp = self.get_file_path_by_key("train_temp_dir_name")
        self.test_temp_dp = self.get_file_path_by_key("test_temp_dir_name")
        self.model_dp = self.get_file_path_by_key("model_dir_name", create_dir=True)

        self.min_num_covered_rxns_by_rct = config_json.get("min_num_covered_rxns_by_rxn_centralized_template")
        self.device = config_json.get("device")
        self.batch_size = config_json.get("batch_size")
        self.epoch_num = config_json.get("epoch_num")
        self.lr_start = config_json.get("lr_start")
        self.lr_end = config_json.get("lr_end")


if __name__ == "__main__":
    pass
