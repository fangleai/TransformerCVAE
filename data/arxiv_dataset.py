import os, random, json, pickle, re
import numpy as np
import torch.utils.data


class ArxivDataset(torch.utils.data.Dataset):
    """
    A dataset for Arxiv
    """

    def __init__(self, texts, preprocess=lambda x: x, sort=False):
        super().__init__()
        self.texts = texts
        self.preprocess = preprocess
        self.sort=sort

        # if self.sort:
        #     self.data = []
        #     for i in range(len(self.texts)):
        #         type, title, story = self.texts[i]
        #
        #         title = type + ' <sep> ' + title.strip()
        #         story = story.strip()
        #         text_raw_dict = {'title': title, 'story': story}
        #
        #         text = self.preprocess(text_raw_dict)
        #         self.data.append(text)
        #     self.data.sort(key=lambda x: len(x[0]), reverse=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        if self.sort:
            return self.data[i]
        else:
            type, title, story = self.texts[i]

            title = type + ' <sep> ' + title.strip()
            story = story.strip()
            text_raw_dict = {'title': title, 'story': story}

            text = self.preprocess(text_raw_dict)
            return text
