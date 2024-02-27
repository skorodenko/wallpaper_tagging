import torch
import polars as pl
import lightning as lg
from models.utils import TagEncoder
from pathlib import Path
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader, random_split


class CustomDataset(Dataset):
    
    def __init__(self, files: list[str] | str, root: str, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = pl.read_ndjson(*files).with_row_count(name="idx")

    def __len__(self):
        return self.images.select(pl.len()).item()

    def __getitem__(self, idx: int):
        row = self.images.filter(pl.col("idx") == idx)
        image_path = Path(
            self.root,
            row["image_root"].item(),
            row["file_name"].item(),
        )
        image = read_image(str(image_path))
        labels = row["labels"].item()
        tags = row["tags"].item()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
            tags = self.target_transform(tags)
        return image, tags, labels


class DataModule(lg.LightningDataModule):
    
    def __init__(self, root: str = "", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.prepare_data_per_node = False
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.Compose([
            TagEncoder(),
            transforms.ToDtype(torch.float32)
        ])
        self.root = Path(root)
        self.train_data = [
            #str(self.root  / "assets" / "preprocessed" / "Train_coco.ndjson"),
            str(self.root / "assets" / "preprocessed" / "Train_nus-wide.ndjson"),
        ]
        self.test_data = [
            #str(self.root / "assets" / "preprocessed" / "Test_coco.ndjson"),
            str(self.root / "assets" / "preprocessed" / "Test_nus-wide.ndjson"),
        ]
    
    def setup(self, stage: str):
        if stage == "fit":
            dataset = CustomDataset(
                self.train_data,
                root = self.root,
                transform = self.transform,
                target_transform = self.target_transform,
            )
            self.train, self.val = random_split(
                dataset=dataset, 
                lengths=[0.7, 0.3], 
                generator=torch.Generator().manual_seed(42)
            )

        if stage == "test":
            self.test = CustomDataset(
                self.test_data,
                root = self.root,
                transform = self.transform,
                target_transform = self.target_transform,
            )

        if stage == "predict":
            self.predict = self.test
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, pin_memory=True)