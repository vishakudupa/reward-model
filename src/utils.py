import torch
from tokenizers import pre_tokenizers
from torch.utils.data import ConcatDataset, random_split
from transformers import AutoTokenizer
import json
import random
from dataset import AnthropicRLFH, HFSummary, WebGPT, KaggleDataset

SPECIAL_TOKENS = {"prompter": "|prompter|", "assistant": "|assistant|"}
generator = torch.Generator().manual_seed(42)


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    if hasattr(config, "per_digit_tokens") and config.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if config.special_tokens:
        special_tokens = {
            "pad_token": config.special_tokens.pad_token,
            "eos_token": config.special_tokens.eos_token,
            "sep_token": config.special_tokens.sep_token,
        }
        tokenizer.add_special_tokens(special_tokens)

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
    )

    return tokenizer


def get_single_dataset(name, is_val=False, **kwargs):
    if name == "hf_summary":
        dataset = HFSummary(**kwargs)
    elif name == "webgpt":
        dataset = WebGPT(**kwargs)
    elif name == "AnthropicRLHF":
        dataset = AnthropicRLFH(**kwargs)
    elif name == "kaggle":
        dataset = KaggleDataset(is_val=is_val)
    else:
        raise ValueError(f"Invalid dataset name {name}")

    return dataset


def prepare_datasets(config):
    with open('dataset_resume_label.json', 'r') as f:
        dataset_resume_label = json.load(f)
    
    # shuffle the dataset with random state 42
    random.Random(42).shuffle(dataset_resume_label)
    # split the dataset into train and validation
    split = int(len(dataset_resume_label) * (1 - config.validation_size))
    print('Splitting dataset into train and validation with split size: ', split)
    train_dataset = dataset_resume_label[:split]
    val_dataset = dataset_resume_label[split+1:]

    # save it as train.json and test.json
    with open('train.json', 'w') as f:
        json.dump(train_dataset, f)
    with open('test.json', 'w') as f:
        json.dump(val_dataset, f)

    name = list(config.datasets[0].keys())[0]
    kwargs = config.datasets[0][name]

    return get_single_dataset(name, is_val=False, **kwargs), get_single_dataset(name, is_val=True, **kwargs)
