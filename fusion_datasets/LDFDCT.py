import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .sr_util import get_paths_from_images, transform_augment


class LDFDCT(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1, hu_min=-1024.0, hu_max=1500.0):
        self.img_size = img_size
        self.data_len = data_len
        self.split = split
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.backend = self._detect_backend(dataroot)

        if self.backend == "hf":
            self.dataset = self._load_hf_dataset(dataroot)
            self.dataset_len = len(self.dataset)
            if self.data_len > 0:
                self.data_len = min(self.data_len, self.dataset_len)
                self.dataset = self.dataset.select(range(self.data_len))
            else:
                self.data_len = self.dataset_len
        else:
            self.img_ld_path, self.img_fd_path = get_paths_from_images(dataroot)
            dataset_len = len(self.img_ld_path)
            if self.data_len <= 0:
                self.data_len = dataset_len
            else:
                self.data_len = min(self.data_len, dataset_len)
                self.img_ld_path = self.img_ld_path[:self.data_len]
                self.img_fd_path = self.img_fd_path[:self.data_len]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.backend == "hf":
            sample = self.dataset[index]
            img_ld = self._prepare_hf_image(sample["qd"])
            img_fd = self._prepare_hf_image(sample["fd"])
            case_name = f"{sample['patient']}_s{int(sample['slice_idx']):04d}"
        else:
            ld_path = self.img_ld_path[index]
            fd_path = self.img_fd_path[index]
            base_name = os.path.basename(ld_path)
            case_name = base_name.split('_')[0]
            img_ld = Image.open(ld_path).convert("L")
            img_fd = Image.open(fd_path).convert("L")
            img_ld = img_ld.resize((self.img_size, self.img_size))
            img_fd = img_fd.resize((self.img_size, self.img_size))

        img_ld, img_fd = transform_augment(
            [img_ld, img_fd], split=self.split, min_max=(-1, 1))

        return {'FD': img_fd, 'LD': img_ld, 'case_name': case_name}

    def _detect_backend(self, dataroot):
        root = Path(dataroot)
        if root.is_dir() and (
            (root / "dataset_info.json").exists() or (root / "state.json").exists()
        ):
            return "hf"
        return "png"

    def _load_hf_dataset(self, dataroot):
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "Install the `datasets` package to read Hugging Face datasets (pip install datasets)."
            ) from exc
        return load_from_disk(str(dataroot))

    def _prepare_hf_image(self, array_2d):
        arr = np.asarray(array_2d, dtype=np.float32)
        arr = np.clip(arr, self.hu_min, self.hu_max)
        arr = (arr - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray(arr)
        if self.img_size is not None:
            img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        return img
