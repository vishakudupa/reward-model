import re
from collections import defaultdict
from typing import List, Union

from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import pandas as pd
import json


class HFSummary(Dataset):
    name = "openai/summarize_from_feedback"

    def __init__(self, split: Union[List[str], str] = "train"):
        super().__init__()
        if isinstance(split, str):
            split = [split]
        if isinstance(split, OmegaConf):
            self.split = OmegaConf.to_object(split)
        else:
            self.split = split
        dataset = load_dataset(self.name, "axis", split=self.split)
        self.data_dict = self.prepare_axis(dataset)
        self.postids = list(self.data_dict.keys())

    def prepare_axis(self, dataset):
        data_dict = defaultdict(dict)
        for data in dataset:
            for item in data:
                if item["summary"]["axes"].get("overall") is not None:
                    postid = item["info"]["id"]
                    summary = {k: item["summary"][k] for k in ["text", "axes"]}
                    if postid not in data_dict.keys():
                        instruction = "summarize: " + (
                            item["info"]["post"] or item["info"]["article"]
                        )
                        data_dict[postid].update(
                            {"post": instruction, "summaries": [summary]}
                        )
                    else:
                        data_dict[postid]["summaries"].append(summary)

        return data_dict

    def __len__(self):
        return len(self.postids)

    def __getitem__(self, idx):
        post, summaries = self.data_dict[self.postids[idx]].values()
        summaries = sorted(summaries, key=lambda x: x["axes"]["overall"], reverse=True)
        dedup_dict = {item["axes"]["overall"]: item["text"] for item in summaries}
        summaries = {key: val for val, key in dedup_dict.items()}
        summaries = list(summaries.keys())
        return [post], summaries

class KaggleDataset:
    name = "custom/kaggle"

    def __init__(self, is_val=False):
        super().__init__()
        if not is_val:
            self.df = pd.read_json('train.json')
        else:
            self.df = pd.read_json('test.json')
        # indices where label is 1
        self.indices_1 = self.df[self.df['label'] == 1].index.tolist()
        # indices where label is 0
        self.indices_0 = self.df[self.df['label'] == 0].index.tolist()

        # print number of samples in each class
        print('Number of samples in class 1:', len(self.indices_1))
        print('Number of samples in class 0:', len(self.indices_0))

        # create pairs of indices
        self.indices_pairs = []
        for i in range(len(self.indices_1)):
            for j in range(len(self.indices_0)):
                self.indices_pairs.append((self.indices_1[i], self.indices_0[j]))

        # print number of pairs
        print('Number of pairs:', len(self.indices_pairs))

    def __len__(self):
        return len(self.indices_pairs)

    def __getitem__(self, idx):
        # returns tuple of two resumes from df
        return self.df.iloc[self.indices_pairs[idx][0]]['resume'], self.df.iloc[self.indices_pairs[idx][1]]['resume']

class WebGPT:
    name = "openai/webgpt_comparisons"

    def __init__(self, split: str = "train"):
        super().__init__()
        self.split = split
        dataset = load_dataset(self.name, split=self.split)
        self.dataset_dict = defaultdict(dict)
        for item in dataset:
            post_id = item["question"]["id"]
            if post_id not in self.dataset_dict.keys():
                self.dataset_dict[post_id] = {
                    "full_text": item["question"]["full_text"],
                    "answers": [],
                }
                if item["score_0"] > 0:
                    answers = [item["answer_0"], item["answer_1"]]
                elif item["score_0"] < 0:
                    answers = [item["answer_1"], item["answer_0"]]
                else:
                    answers = []
                answers = [re.sub(r"\[\d+\]", "", answer) for answer in answers]
                answers = [
                    ".".join([sent.strip() for sent in answer.split(".")])
                    for answer in answers
                ]
                if answers:
                    self.dataset_dict[post_id]["answers"].extend(answers)
                else:
                    _ = self.dataset_dict.pop(post_id)

        self.post_ids = list(self.dataset_dict.keys())

    def __len__(self):
        return len(self.post_ids)

    def __getitem__(self, idx):
        question, answers = self.dataset_dict[self.post_ids[idx]].values()
        return [question], answers


class AnthropicRLFH(Dataset):
    name = "Dahoas/full-hh-rlhf"

    def __init__(self, split: Union[List[str], str] = "train"):
        super().__init__()
        if isinstance(split, str):
            split = [split]
        if isinstance(split, OmegaConf):
            self.split = OmegaConf.to_object(split)
        else:
            self.split = split
        dataset = load_dataset(self.name, split=self.split)
        self.data_dict = defaultdict(dict)
        id = 0
        for data in dataset:
            for item in data:
                dialogs = [
                    text.replace("\n\n", "").strip()
                    for text in re.split(r"Human:|Assistant:", item["prompt"])
                ]
                dialogs = [text for text in dialogs if text != ""]
                self.data_dict[f"prompt{id}"].update(
                    {"prompt": dialogs, "answers": [item["chosen"], item["rejected"]]}
                )
                id += 1

        self.prompt_ids = list(self.data_dict.keys())

    def __len__(
        self,
    ):
        return len(self.prompt_ids)

    def __getitem__(self, idx):
        prompt, answers = self.data_dict.get(self.prompt_ids[idx], {}).values()
        return prompt, answers
