import polars as pl
import polars.selectors as cs


OUTPUT = "./assets/nus-wide/Test_nus-wide.ndjson"
IMAGES = "./assets/nus-wide/TestImagelist.txt"
TAGS_81 = "./assets/nus-wide/Test_Tags81.txt"
TAGS_1K = "./assets/nus-wide/Test_Tags1k.dat"

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

images = tags_1k.join(tags_81, on="file_name")

images.collect().write_ndjson(OUTPUT)
