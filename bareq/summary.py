"""Print a short summary of a bareq-data folder (to paste back after the Kaggle run).

  python -m bareq.summary /kaggle/working/bareq-data
"""
import json
import os
import sys


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/bareq-data"
    m = json.load(open(os.path.join(d, "manifest.json")))
    r = m["lol"]["dedupe"]
    print("commit:", m["git_commit"][:7], "| seconds:", m["seconds"], "| cpus/workers:", m["cpus"], m["workers"])
    print("LOL found:", r["found"], "| v1 photos copied in v2:", r["v1_images_copied_in_v2"])
    print("LOL kept:", r["kept"], "| dropped:", r["dropped_counts"])
    print("LOL histogram, train -> nearest test (0-1.5, step 0.1):", r["hist_train_to_nearest_test"])
    print("LOL histogram, test -> nearest other test:", r["hist_test_to_nearest_other_test"])
    for sp in ("train", "val", "test"):
        s = m["sidd"][sp]
        print(f"SIDD {sp}: scenes {s['scenes']}, {len(s['instances'])} images, {s['crops']} crops")
    v = m["div2k"]
    print("DIV2K train/val/test:", len(v["train"]), len(v["val"]), len(v["test"]), "| train patches:", v["train_patches"])
    print("leak check:", m["leak_check"])
    for f in sorted(os.listdir(d)):
        print(f"  {os.path.getsize(os.path.join(d, f)) / 1e6:9.1f} MB  {f}")


if __name__ == "__main__":
    main()
