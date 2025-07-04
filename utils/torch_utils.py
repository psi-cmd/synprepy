#!/usr/bin/env python 

import torch


class TorchUtils:

    @classmethod
    def label_idx_to_one_hot(cls, label_idx, num_class, device):
        # row, col = label_idx.shape
        # tensor = label_idx.reshape(col, row)[0]
        row = label_idx.shape[0]
        one_hot = torch.zeros(row, num_class).long().to(device)
        one_hot.scatter_(dim=1, index=label_idx.unsqueeze(dim=1), src=torch.ones(row, num_class).long().to(device))
        return one_hot


if __name__ == "__main__":
    pass
