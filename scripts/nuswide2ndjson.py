import glob
import polars as pl
import polars.selectors as cs
from tqdm import tqdm
from pathlib import Path
from torchvision.io import read_image


IMAGE_ROOT = "./assets/nus-wide/images"
OUTPUT = "./assets/preprocessed/Train_nus-wide.ndjson"
IMAGES = "./assets/nus-wide/TrainImagelist.txt"
TAGS_81 = "./assets/nus-wide/Train_Tags81.txt"
TAGS_1K = "./assets/nus-wide/Train_Tags1k.dat"


IMAGE_FILES = glob.glob(r"*.*", root_dir=IMAGE_ROOT)


header_1k = pl.read_csv("./assets/nus-wide/TagList1k.txt", has_header=False)["column_1"].to_list()
header_81 = pl.read_csv("./assets/nus-wide/TagList81.txt", has_header=False)["column_1"].to_list()

tags_1k = pl.scan_csv(
    TAGS_1K, 
    separator="\t",
    has_header=False,
    new_columns=header_1k,
).drop(cs.string())

tags_81 = pl.scan_csv(
    TAGS_81, 
    separator=" ",
    has_header=False,
    new_columns=header_81,
).drop(cs.string())

images = pl.scan_csv(
    IMAGES,
    has_header=False,
    new_columns=["file_name"],
)

tags_1k = pl.concat([images, tags_1k], how="horizontal")
tags_1k = (
    tags_1k.melt(
        id_vars="file_name",
        value_vars=header_1k,
        variable_name="labels",
    )
    .filter(pl.col("value") != 0)
    .drop("value")
)
tags_1k = tags_1k.group_by("file_name").agg(pl.col("labels"))

tags_81 = pl.concat([images, tags_81], how="horizontal")
tags_81 = (
    tags_81.melt(
        id_vars="file_name",
        value_vars=header_81,
        variable_name="tags",
    )
    .filter(pl.col("value") != 0)
    .drop("value")
)
tags_81 = tags_81.group_by("file_name").agg(pl.col("tags"))

print(tags_1k.collect())
print(tags_81.collect())

images = tags_81.join(tags_1k, on="file_name", how="left")


# Add root path
images = images.with_columns(pl.lit(IMAGE_ROOT).alias("image_root"))

# Before file check up
print(images.collect())

# Checks image existance
images = images.filter(pl.col("file_name").is_in(IMAGE_FILES))

# Checks if images are not corrupted (by trying to read)
dflen = images.select(pl.count("file_name")).collect().item()
print("Starting check for image corruption")
dellist = []
for row in tqdm(
    images.select([pl.col("image_root"), pl.col("file_name")]).collect().to_dicts()
):
    img = Path(row["image_root"], row["file_name"])
    try:
        read_image(str(img))
    except RuntimeError:
        dellist.append(row["file_name"])

images = images.filter(~pl.col("file_name").is_in(dellist))

# After file checkup
print(images.collect())

images.collect().write_ndjson(OUTPUT)
