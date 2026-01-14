import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from loguru import logger


class TextPromptDataset(Dataset):
    def __init__(self, dataset: str, split: str = "train"):
        """Load plain text prompts for train/test splits.

        Args:
            dataset: Root dataset directory.
            split: File prefix (e.g., `train` or `test`).
        """
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int | Tuple[int, int]) -> Dict:
        """Return a prompt item, carrying sampler epoch_tag if provided."""
        epoch_tag = None
        if isinstance(idx, tuple):
            epoch_tag, idx = idx
        return {"epoch": epoch_tag, "prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples: List[Dict]) -> Tuple[Optional[int], List[str], List[Dict]]:
        """Batch prompts while preserving a consistent epoch tag."""
        epoch_tags = [example.get("epoch") for example in examples]
        epoch_tag = (
            epoch_tags[0] if all(tag == epoch_tags[0] for tag in epoch_tags) else None
        )
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return epoch_tag, prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset: str, split: str = "train"):
        """Load Geneval prompts with metadata for the given split."""
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int | Tuple[int, int]) -> Dict:
        """Return a prompt+metadata, carrying sampler epoch_tag if provided."""
        epoch_tag = None
        if isinstance(idx, tuple):
            epoch_tag, idx = idx
        return {
            "epoch": epoch_tag,
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx],
        }

    @staticmethod
    def collate_fn(examples: List[Dict]) -> Tuple[Optional[int], List[str], List[Dict]]:
        """Batch Geneval items while preserving epoch tags."""
        epoch_tags = [example.get("epoch") for example in examples]
        epoch_tag = (
            epoch_tags[0] if all(tag == epoch_tags[0] for tag in epoch_tags) else None
        )
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return epoch_tag, prompts, metadatas


class JsonPromptDataset(Dataset):
    """Load prompts from JSON files (one JSON object per line).

    Optimized for large datasets by using lazy loading and efficient file reading.
    """

    def __init__(self, dataset: str, split: str = "train"):
        """Load prompts from JSON files.

        Args:
            dataset: Root dataset directory.
            split: File prefix (e.g., `train` or `test`).

        Returns:
            None
        """
        self.file_path = os.path.join(dataset, f"{split}.json")
        self._prompts = None
        self._metadatas = None
        self._file_size = (
            os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
        )

        # Optimization strategy:
        # - For training data, load directly to memory even if large (frequent random access needed)
        # - Use lazy loading only for very large files (>1GB)
        # - For test data, usually small, load directly
        if self._file_size > 1024 * 1024 * 1024:  # 1GB
            # Very large file: build index, lazy loading
            self._use_lazy_loading = True
            self._line_offsets = []
            self._load_line_offsets()
        else:
            # Small/medium files: load directly to memory (more efficient)
            self._use_lazy_loading = False
            self._load_all_prompts()

    def _load_line_offsets(self):
        """Build offset index for each line for fast lookup (only for very large files).

        Args:
            None

        Returns:
            None
        """
        logger.info(f"Building index for very large file: {self.file_path}")
        self._line_offsets = [0]  # First line starts at 0
        with open(self.file_path, "rb") as f:
            current_offset = 0
            while True:
                chunk = f.read(8192)  # 8KB chunks
                if not chunk:
                    break
                chunk_start = current_offset
                current_offset += len(chunk)
                # Find all newlines in chunk
                pos = 0
                while True:
                    pos = chunk.find(b"\n", pos)
                    if pos == -1:
                        break
                    # Calculate absolute position of newline in file
                    line_start = chunk_start + pos + 1
                    self._line_offsets.append(line_start)
                    pos += 1

        logger.info(f"Index building completed, {len(self._line_offsets)} lines")

    def _load_all_prompts(self):
        """Load all prompts directly to memory (for small files).

        Args:
            None

        Returns:
            None
        """
        self._prompts = []
        self._metadatas = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        prompt = item.get("prompt", "")
                        if prompt:
                            self._prompts.append(prompt)
                            # Save complete metadata (may contain other fields)
                            metadata = {k: v for k, v in item.items() if k != "prompt"}
                            self._metadatas.append(metadata)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
                        continue

    def _get_item_lazy(self, idx: int) -> Dict:
        """Lazy loading: read data for specified line from file.

        Args:
            idx: Index of the line to read.

        Returns:
            Dict containing "prompt" and "metadata" keys.
        """
        if idx >= len(self._line_offsets):
            raise IndexError(f"Index {idx} out of range")

        start_offset = self._line_offsets[idx]
        # Calculate end offset (start of next line or end of file)
        end_offset = (
            self._line_offsets[idx + 1]
            if idx + 1 < len(self._line_offsets)
            else self._file_size
        )

        with open(self.file_path, "r", encoding="utf-8") as f:
            f.seek(start_offset)
            line = f.read(end_offset - start_offset).strip()
            if line:
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", "")
                    metadata = {k: v for k, v in item.items() if k != "prompt"}
                    return {"prompt": prompt, "metadata": metadata}
                except json.JSONDecodeError:
                    return {"prompt": "", "metadata": {}}
        return {"prompt": "", "metadata": {}}

    def __len__(self) -> int:
        """Get the number of items in the dataset.

        Args:
            None

        Returns:
            Number of items in the dataset.
        """
        if self._use_lazy_loading:
            return len(self._line_offsets)
        return len(self._prompts) if self._prompts else 0

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """Return a prompt item, carrying sampler epoch_tag if provided.

        Args:
            idx: Index or tuple of (epoch_tag, idx) for the item to retrieve.

        Returns:
            Dict containing "epoch", "prompt", and "metadata" keys.
        """
        epoch_tag = None
        if isinstance(idx, tuple):
            epoch_tag, idx = idx

        if self._use_lazy_loading:
            item = self._get_item_lazy(idx)
        else:
            item = {
                "prompt": self._prompts[idx],
                "metadata": self._metadatas[idx] if self._metadatas else {},
            }

        return {
            "epoch": epoch_tag,
            "prompt": item["prompt"],
            "metadata": item["metadata"],
        }

    @staticmethod
    def collate_fn(examples: List[Dict]) -> Tuple[Optional[int], List[str], List[Dict]]:
        """Batch prompts while preserving a consistent epoch tag.

        Args:
            examples: List of example dictionaries, each containing "epoch", "prompt", and "metadata".

        Returns:
            Tuple of (epoch_tag, prompts, metadatas) where epoch_tag is None if inconsistent.
        """
        epoch_tags = [example.get("epoch") for example in examples]
        epoch_tag = (
            epoch_tags[0] if all(tag == epoch_tags[0] for tag in epoch_tags) else None
        )
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return epoch_tag, prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        """Repeat each prompt k times per global batch and shard across ranks.

        Args:
            dataset: Dataset to sample from.
            batch_size: Per-rank batch size.
            k: Repetition factor per prompt.
            num_replicas: World size.
            rank: Current rank id.
            seed: Base seed for deterministic shuffles.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        """Yield per-rank batches with (epoch_tag, idx) pairs."""
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(
                    [(self.epoch, idx) for idx in shuffled_samples[start:end]]
                )
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch: int):
        """Set epoch tag to keep RNG in sync across workers."""
        self.epoch = epoch


def build_dataloaders(
    cfg, accelerator
) -> Tuple[DataLoader, DataLoader, DistributedKRepeatSampler]:
    """Construct train/eval dataloaders and sampler with epoch tags.

    Args:
        cfg: Parsed training configuration.
        accelerator: Accelerator instance to read rank/world info.

    Returns:
        Tuple of (train_dataloader, test_dataloader, train_sampler).
    """
    if cfg.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(cfg.paths.dataset, "train")
        test_dataset = TextPromptDataset(cfg.paths.dataset, "test")
        collate_fn = TextPromptDataset.collate_fn
    elif cfg.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(cfg.paths.dataset, "train")
        test_dataset = GenevalPromptDataset(cfg.paths.dataset, "test")
        collate_fn = GenevalPromptDataset.collate_fn
    elif cfg.prompt_fn == "filtered_prompts":
        # Load JSON format prompts (one JSON object per line)
        train_dataset = JsonPromptDataset(cfg.paths.dataset, "train")
        test_dataset = JsonPromptDataset(cfg.paths.dataset, "test")
        collate_fn = JsonPromptDataset.collate_fn
    else:
        raise NotImplementedError(
            "Only general_ocr, geneval, or filtered_prompts prompt_fn supported"
        )


    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=cfg.sample.batch_size,
        k=cfg.sample.num_video_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=cfg.seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=collate_fn,
        prefetch_factor=1,
        persistent_workers=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.sample.eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=8,
    )
    return train_dataloader, test_dataloader, train_sampler
