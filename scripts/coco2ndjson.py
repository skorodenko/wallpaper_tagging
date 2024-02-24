import json
import polars as pl


OUTPUT = "./assets/coco/Test_coco.ndjson"
FILE = "./assets/coco/annotations/instances_val2017.json"


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

images.write_ndjson(OUTPUT)
