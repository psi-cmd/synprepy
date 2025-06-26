#!/usr/bin/env python 

import os
import json
import logging

from config.rcts_config import RCTSConfig
from config.config_base import ConfigBase


class Config(ConfigBase):

    def __init__(self, config_json_fp):
        with open(config_json_fp, 'r', encoding='utf-8')as f:
            _config_json = json.load(f)
        super(Config, self).__init__(_config_json)

        self.rxn_data_tsv_file_path = self.get_file_path_by_key("rxn_data_tsv_file_name", True)
        self.rid_with_rxn_template_tsv_file_path = self.get_file_path_by_key("rid_with_rxn_template_tsv_file_name")
        self.rxn_centralized_template_tsv_file_path = self.get_file_path_by_key("rxn_centralized_template_tsv_file_name")
        self.rxn_extended_template_tsv_file_path = self.get_file_path_by_key("rxn_extended_template_tsv_file_name")
        self.rid_with_tid_tsv_file_path = self.get_file_path_by_key("rid_with_tid_tsv_file_name")

        self.rcts_config = RCTSConfig(_config_json.get("rxn_centralized_template_selector_config"),
                                      self.rxn_data_tsv_file_path,
                                      self.rxn_centralized_template_tsv_file_path,
                                      self.rid_with_tid_tsv_file_path)


if __name__ == "__main__":
    pass
