import polars as pl

coco = pl.scan_ndjson("./assets/preprocessed/Train_coco.ndjson")
nuswide = pl.scan_ndjson("./assets/preprocessed/Train_nus-wide.ndjson")

#coco = coco.with_columns(
#    ext = pl.col("file_name").str.extract(r"\.(.*)$"),
#    iid = pl.col("file_name").str.extract(r"[^\..]+$")
#)
#nuswide = nuswide.with_columns(
#    ext = pl.col("file_name").str.extract(r"\.(.*)"),
#    iid = pl.col("file_name").str.extract(r"[^.\.]+$")
#)
#print(coco.collect()["ext"].unique())
#print(coco.collect()["iid"].unique())
#print(nuswide.collect()["ext"].unique().sort())
#print(nuswide.collect()["iid"].unique().sort())
print(coco.collect())
print(coco.filter(pl.col("file_name").str.contains(r"\.")).collect())
print(nuswide.collect())
print(nuswide.filter(pl.col("file_name").str.contains(r"\.")).collect())