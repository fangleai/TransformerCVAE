import os, random, json, pickle, re
import numpy as np
import torch.utils.data


class YelpDataset(torch.utils.data.Dataset):
    """
    A dataset for Yelp
    """

    def __init__(self, source, preprocess=lambda x: x, sort=False):
        super().__init__()
        self.preprocess = preprocess
        self.sort=sort

        print('Loading Yelp...')
        with open(source, errors='ignore') as fs:
            self.source = fs.readlines()
        print('Done.')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        raw = self.source[i]
        title, story = raw[:1], raw[2:].strip()
        text_raw_dict = {'title': title, 'story': story}

        text = self.preprocess(text_raw_dict)
        return text
