import polars as pl


FILES = [
    "/home/rinkuro/Sandbox/wallpaper_app/wallpaper_tagging/assets/preprocessed/Train_nus-wide.ndjson",
]

OTAGS = "./assets/preprocessed/Tags_nus-wide.ndjson"
OLABELS = "./assets/preprocessed/Labels_nus-wide.ndjson"

df = pl.scan_ndjson(FILES)

tags = df.select(pl.col("tags")).explode("tags")
labels = df.select(pl.col("labels")).explode("labels")

tags = tags.select(pl.col("tags").value_counts())
labels = labels.select(pl.col("labels").value_counts())

tags = tags.unnest("tags").rename({"tags": "name"})
labels = labels.unnest("labels").rename({"labels": "name"})

tags = tags.sort(pl.col("count"), descending=True)
labels = labels.sort(pl.col("count"), descending=True)

print(tags.collect())
print(labels.collect())
tags.collect().write_ndjson(OTAGS)
labels.collect().write_ndjson(OLABELS)