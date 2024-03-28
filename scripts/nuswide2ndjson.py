import glob
import polars as pl
from tqdm import tqdm
from pathlib import Path
from torchvision.io import read_image


IMAGE_ROOT = "./assets/nus-wide/images"
OUTPUT = "./assets/preprocessed/Test_nus-wide.ndjson"
IMAGES = "./assets/nus-wide/TestImagelist.txt"
TAGS_1K = "./assets/nus-wide/Test_Tags1k.dat"


LABEL_FILES = glob.glob(r"./assets/nus-wide/src/Groundtruth/*_Test.txt")
IMAGE_FILES = glob.glob(r"*.*", root_dir=IMAGE_ROOT)

LABEL_FILES.sort()

header_1k = pl.read_csv("./assets/nus-wide/TagList1k.txt", has_header=False)["column_1"].to_list()
header_81 = pl.read_csv("./assets/nus-wide/TagList81.txt", has_header=False)["column_1"].to_list()

tags_1k = pl.scan_csv(
    TAGS_1K, 
    separator="\t",
    has_header=False,
    new_columns=header_1k,
)

labels_81 = pl.concat(
    [
        pl.scan_csv(f, has_header=False, new_columns=[header_81[i]]) for i, f in enumerate(LABEL_FILES)
    ], 
    how = "horizontal",
)


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
        variable_name="tags",
    )
    .filter(pl.col("value") != 0)
    .drop("value")
)
tags_1k = tags_1k.group_by("file_name").agg(pl.col("tags"))

labels_81 = pl.concat([images, labels_81], how="horizontal")
labels_81 = (
    labels_81.melt(
        id_vars="file_name",
        value_vars=header_81,
        variable_name="labels",
    )
    .filter(pl.col("value") != 0)
    .drop("value")
)
labels_81 = labels_81.group_by("file_name").agg(pl.col("labels"))


# Conjoined table
images = labels_81.join(tags_1k, on="file_name")


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
    path = Path(row["image_root"], row["file_name"])
    try:
        read_image(str(path))
    except RuntimeError:
        dellist.append(row["file_name"])

images = images.filter(~pl.col("file_name").is_in(dellist))

# After file checkup
print(images.collect())

images.collect().write_ndjson(OUTPUT)
