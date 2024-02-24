import json
import glob
import polars as pl


IMAGE_ROOT = "./assets/coco/images/train2017"
OUTPUT = "./assets/preprocessed/Train_coco.ndjson"
FILE = "./assets/coco/annotations/instances_train2017.json"


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

# After file checkup
print(images)

# Save to disk
images.write_ndjson(OUTPUT)
