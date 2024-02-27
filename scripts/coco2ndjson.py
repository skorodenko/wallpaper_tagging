import json
import glob
import polars as pl
from tqdm import tqdm
from pathlib import Path
from torchvision.io import read_image


IMAGE_ROOT = "./assets/coco/images/val2017"
OUTPUT = "./assets/preprocessed/Test_coco.ndjson"
FILE = "./assets/coco/annotations/instances_val2017.json"


IMAGE_FILES = glob.glob(r"*.*", root_dir=IMAGE_ROOT)


with open(FILE) as f:
    instances_file = json.load(f)

images = pl.DataFrame(
    instances_file["images"],
    schema = {
        "id": pl.Int64,
        "file_name": pl.String,
    }
)

categories = pl.DataFrame(
    instances_file["categories"],
    schema = {
        "id": pl.Int64,
        "name": pl.Utf8,
        "supercategory": pl.Utf8,
    }
)

annotations = pl.DataFrame(
    instances_file["annotations"],
    schema = {
        "image_id": pl.Int64,
        "category_id": pl.Int64,
    }
)

annotations = annotations.join(
    categories, 
    left_on="category_id",
    right_on="id",
)

annotations = (
    annotations
        .group_by("image_id")
        .agg(
            pl.col("name").unique(),
            pl.col("supercategory").unique()
        )
)

images = images.join(
    annotations,
    left_on="id",
    right_on="image_id"
)

images = (
    images
        .drop("id")
        .rename({
            "name": "labels",
            "supercategory": "tags",
        })
)

# Add root path
images = images.with_columns(pl.lit(IMAGE_ROOT).alias("image_root"))

# Before file check up
print(images)

# Checks image existance
images = images.filter(pl.col("file_name").is_in(IMAGE_FILES))

# Checks if images are not corrupted (by trying to read)
dflen = images.select(pl.count("file_name")).item()
print("Starting check for image corruption")
dellist = []
for row in tqdm(
    images.select([pl.col("image_root"), pl.col("file_name")]).to_dicts()
):
    img = Path(row["image_root"], row["file_name"])
    try:
        read_image(str(img))
    except RuntimeError:
        dellist.append(row["file_name"])

images = images.filter(~pl.col("file_name").is_in(dellist))

# After file checkup
print(images)

# Save to disk
images.write_ndjson(OUTPUT)
