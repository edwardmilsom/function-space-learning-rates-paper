import torch
from datasets import load_dataset
import numpy as np
from warnings import warn
from torch.utils.data import Dataset

class PretrainingDataset(Dataset):
    def __init__(self, tokenizer, dataset_name="c4", subset="en", streaming=True, SEQ_LEN=None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.SEQ_LEN = SEQ_LEN

        # Load dataset
        if dataset_name == "cold_french_law":
            self.dataset = load_dataset('harvard-lil/cold-french-law', split='train', streaming=True)
            self.dataset = self.dataset.take(20000)
        elif dataset_name == 'mathpile':
            ###https://huggingface.co/datasets/GAIR/MathPile
            self.dataset = load_dataset('json', data_files='data/mathpile/math_arXiv_v0.2_chunk_1.jsonl', split='train', streaming=True)
            self.dataset = self.dataset.take(20000)
        elif dataset_name == "the_pile":
            self.dataset = load_dataset("EleutherAI/pile", streaming=streaming, split="train")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        buffer_tokens = []
        for example in self.dataset:
            if 'text' in example:
                text = example['text']
            elif 'content' in example:
                text = example['content']
            elif self.dataset_name == 'cold_french_law':
                try:
                    text = f"{example['texte_contexte']} {example['article_contenu_markdown']}"
                except:
                    warn(f"Skipping french example ")
            else:
                raise ValueError("Unknown text field")
            ntoken = example['token_count'] if 'token_count' in example else len(text.split())
            if ntoken < self.SEQ_LEN: continue

            # Tokenize without padding or truncation
            tokens = self.tokenizer(text, truncation=False, padding=False)["input_ids"]
            buffer_tokens.extend(tokens)

            while len(buffer_tokens) >= self.SEQ_LEN:
                yield torch.tensor(buffer_tokens[:self.SEQ_LEN])
                buffer_tokens = buffer_tokens[self.SEQ_LEN:]

    def __getitem__(self, idx):
        # This is a placeholder since we're using streaming
        # Real implementation would need proper indexing if not streaming
        raise NotImplementedError("Use streaming iteration instead of indexing")

def create_pretrain_dataloader(
    tokenizer,
    batch_size=8,
    dataset_name="c4",
    subset="en",
    SEQ_LEN=None,
):
    """Creates a dataloader for pretraining"""

    def collate_fn(examples):
        # Stack the pre-chunked sequences
        batch = torch.stack(examples)

        # Create attention masks
        attention_mask = torch.ones_like(batch)
        attention_mask[batch == tokenizer.pad_token_id] = 0

        # For casual language modeling, labels are the same as inputs
        labels = batch.clone()

        return {
            "input": batch,
            "mask": attention_mask,
            "target": labels
        }

    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        subset=subset,
        SEQ_LEN=SEQ_LEN
    )

    # Create an infinite loader that yields batches
    def infinite_loader():
        while True:
            buffer = []
            for item in dataset:
                buffer.append(item)
                if len(buffer) == batch_size:
                    yield collate_fn(buffer)
                    buffer = []

    return infinite_loader()
