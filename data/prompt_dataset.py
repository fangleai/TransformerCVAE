import os, random, json, pickle, re
import numpy as np
import torch.utils.data


class PromptDataset(torch.utils.data.Dataset):
    """
    A dataset for Writing Prompts
    """

    def __init__(self, source, target, preprocess=lambda x: x, sort=False):
        super().__init__()
        self.preprocess = preprocess
        self.sort=sort

        print('Loading writing prompts...')
        with open(source, errors='ignore') as fs:
            with open(target, errors='ignore') as ft:
                self.prompts = list(zip(fs.readlines(), ft.readlines()))
        print('Done.')

        # if self.sort:
        #     self.data = []
        #     for i in range(len(self.prompts)):
        #         prompt, story = self.prompts[i]
        #
        #         # Remove extra annotation on prompt from WP dataset
        #         prompt = re.sub('\[ (.*) \]', '', prompt)
        #         prompt = prompt.strip()
        #         story = story.strip()
        #         text_raw_dict = {'prompt': prompt, 'story': story}
        #
        #         text = self.preprocess(text_raw_dict)
        #         self.data.append(text)
        #     self.data.sort(key=lambda x: len(x[0]), reverse=True)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        if self.sort:
            return self.data[i]
        else:
            prompt, story = self.prompts[i]

            # Remove extra annotation on prompt from WP dataset
            #prompt = re.sub('\[ (.*) \]', '', prompt)
            prompt = prompt.strip()
            story = story.strip()
            text_raw_dict = {'prompt': prompt, 'story': story}

            text = self.preprocess(text_raw_dict)
            return text
