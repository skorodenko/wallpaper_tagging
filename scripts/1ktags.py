import polars as pl


FILES = [
    "/home/rinkuro/Sandbox/wallpaper_app/wallpaper_tagging/assets/preprocessed/Train_coco.ndjson",
    "/home/rinkuro/Sandbox/wallpaper_app/wallpaper_tagging/assets/preprocessed/Train_nus-wide.ndjson",
]

OUTPUT = "./assets/preprocessed/Tags1k.ndjson"


df = pl.scan_ndjson(FILES)


tags = df.select(pl.col("tags")).explode("tags")
labels = df.select(pl.col("labels")).explode("labels")

tags = tags.select(pl.col("tags").value_counts())
labels = labels.select(pl.col("labels").value_counts())

tags = tags.unnest("tags").rename({"tags": "name"})
labels = labels.unnest("labels").rename({"labels": "name"})

unified = pl.concat([tags, labels])
unified = unified.group_by(pl.col("name")).agg(pl.col("count").sum())
unified = unified.sort(pl.col("count"), descending=True)
unified = unified.head(1000)

print(unified.collect())
unified.collect().write_ndjson(OUTPUT)