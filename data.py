import torch
import pandas as pd
import lightning as lg
from pathlib import Path
from models.utils import TagTransform
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    
    def __init__(self, files: list[str] | str, root: str, load_images: bool = True, transform=None, tag_transform=None, label_transform=None):
        self.root = root
        self.load_images = load_images
        self.transform = transform
        self.tag_transform = tag_transform
        self.label_transform = label_transform
        self.images = pd.read_json(files[0], lines=True)
        if len(files) > 1:
            for f in files[1:]:
                tmp = pd.read_json(f, lines=True)
                self.images = pd.concat([self.images, tmp], ignore_index=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        row = self.images.iloc[idx]
        image_path = Path(
            self.root,
            row["image_root"],
            row["file_name"],
        )
        if self.load_images:
            image = read_image(str(image_path))
        else:
            image = 0
        labels = row["labels"]
        tags = row["tags"]
        if self.load_images and self.transform:
            image = self.transform(image)
        if self.label_transform:
            labels = self.label_transform(labels)
        if self.tag_transform:
            tags = self.tag_transform(tags)
        return image, tags, labels


class DataModule(lg.LightningDataModule):
    
    def __init__(self, root: str = "", load_images: bool = True, batch_size: int = 32, prefetch_factor: int = None, num_workers: int = 0):
        super().__init__()
        self.load_images = load_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prepare_data_per_node = False
        self.transform = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.root = Path(root)
        self.dataroot = self.root / "assets" / "preprocessed"
        self.tag_transform = transforms.Compose([
            TagTransform(self.dataroot / "Tags_nus-wide.ndjson"),
            transforms.ToDtype(torch.float32)
        ])
        self.label_transform = transforms.Compose([
            TagTransform(self.dataroot / "Labels_nus-wide.ndjson"),
            transforms.ToDtype(torch.float32)
        ])
        self.train_data = [
            str(self.dataroot / "Train_nus-wide.ndjson"),
        ]
        self.test_data = [
            str(self.dataroot / "Test_nus-wide.ndjson"),
        ]
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train = CustomDataset(
                self.train_data,
                root = self.root,
                load_images = self.load_images,
                transform = self.transform,
                label_transform = self.label_transform,
                tag_transform = self.tag_transform,
            )
            self.val = CustomDataset(
                self.test_data,
                root = self.root,
                load_images = self.load_images,
                transform = self.transform,
                label_transform = self.label_transform,
                tag_transform = self.tag_transform,
            )

        if stage == "test":
            self.test = CustomDataset(
                self.test_data,
                root = self.root,
                load_images = self.load_images,
                transform = self.transform,
                label_transform = self.label_transform,
                tag_transform = self.tag_transform,
            )

        if stage == "predict":
            self.predict = self.test
            
    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, 
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict, batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True,
            prefetch_factor=self.prefetch_factor
        )