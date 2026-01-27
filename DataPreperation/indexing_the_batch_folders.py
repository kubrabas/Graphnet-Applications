from pathlib import Path
import re

def reindex_split(old_split_dir: str, new_split_dir: str):
    old = Path(old_split_dir)
    new = Path(new_split_dir)
    (new / "truth").mkdir(parents=True, exist_ok=True)
    (new / "features").mkdir(parents=True, exist_ok=True)

    rx = re.compile(r"_(\d+)\.parquet$")
    ids = sorted(int(rx.search(p.name).group(1)) for p in (old/"truth").glob("truth_*.parquet"))

    for new_id, old_id in enumerate(ids):
        for table in ["truth", "features"]:
            src = old / table / f"{table}_{old_id}.parquet"
            dst = new / table / f"{table}_{new_id}.parquet"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)

# reindex_split(
#     "/project/def-nahee/kbas/POM_Response_Parquet/merged/test",
#     "/project/def-nahee/kbas/POM_Response_Parquet/merged/test_reindexed",
# )

