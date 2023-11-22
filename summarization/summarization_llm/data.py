import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
import numpy as np
import datasets
from datasets import dataset_dict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
import re
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Any
from pathlib import Path
import os


class DataModule(pl.LightningDataModule):
    def __init__(self,
    prompt: str = "summarize:",
    model_path: Any = None,
    size: Any = None,
    batch_size: int = 2,
    seq_length: int = 1024,
    overwrite: bool = False) -> None:
        super().__init__()
    def __post_init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def build_prompt(self, example):
        example[""] =  f"""<s>[INST] <<SYS>>
            {self.prompt}
            <</SYS>>
            Summarize this Text: {example["document"]} [/INST]"""
        return example
        

    def build_prompt_simple(self, example):
        example["document"] =  f"""
        Summarize this Text:
        {example["document"]}
        ---
        Summary: 
        """
        return example
    def prepare_data(self):

        mod_path = Path(__file__).parent
        if not os.path.exists(f"{mod_path}/data") or self.overwrite:
            data = load_dataset("xsum")
            if self.size:
                for split_name, split_data  in data.items():
                    data[split_name] = split_data.select(range(self.size))
            data = data.map(self.build_prompt_simple, batched=False)
            data = self.tokenize_data(data)
            data.save_to_disk(f"{mod_path}/data/sumdata5.hf")

    def tokenize_data(self, data):
        def tokenize(example):
            model_inputs = self.tokenizer(
                example["document"],
                padding="max_length",
                max_length=self.seq_length,
                truncation=True,
                return_tensors="pt",
            )
            labels = self.tokenizer(
                example["summary"],
                padding="max_length",
                max_length=self.seq_length,
                truncation=True,
                return_tensors="pt",
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        data = data.map(tokenize, batched=True)
        return data



    def setup(self, stage: str):
        mod_path = Path(__file__).parent
        data = datasets.load_from_disk(f"{mod_path}/data/sumdata5.hf")
        data = data.remove_columns(["document", "summary", "id"])
        data.set_format("pt")
        match stage:
            case "fit":
                self.train = data["train"]
                self.validation = data["validation"]
            case "validation":
                self.validation = data["validation"]
            case "test":
                self.test = data["test"]
            case _:
                raise ValueError("invalid stage")
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)