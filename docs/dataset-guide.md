# Dataset guide: D-Fire

FireWatch-AI is developed against **D-Fire**, a public image dataset for fire and
smoke detection by Gaia, solutions on demand. Datasets are **never** stored in
this Git repository.

## What the upstream Git repository is (and is not)

`https://github.com/gaia-solutions-on-demand/DFireDataset` contains only:

| File | Content |
| --- | --- |
| `README.md` | Description, class/box counts, download links, citations |
| `LICENSE` | CC0-1.0 |
| `utils/utils.py` | `yolo2pixel` / `pixel2yolo` coordinate helpers |
| `figures/dfire_examples.png` | Example image grid (~11 MB) |

It does **not** contain images or labels. A local clone of it (commit
`4bf9c31`) sits at `./DFireDataset/` in the developer's working copy. It is
ignored by Git and is kept only as reference. The coordinate helpers were
adapted into `src/firewatch/data/yolo.py` with attribution.

To reproduce that reference clone (optional):

```bash
git clone https://github.com/gaia-solutions-on-demand/DFireDataset.git DFireDataset
git -C DFireDataset checkout 4bf9c31
```

## Getting the actual data

The upstream README lists these sources (check it for current links):

- D-Fire images + labels (OneDrive, linked from the upstream README)
- Pre-split training / validation / test sets (OneDrive, linked from the upstream README)
- Kaggle mirror: <https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo>

Downloads go **outside** the repository, for example:

- Local (Windows): `D:\datasets\dfire`
- Cluster: `~/datasets/dfire`

The compute nodes may not have internet access, so download on your laptop or the
login node. Then copy the data into the cluster home or project storage, e.g.
`scp -r dfire <user>@<cluster>:~/datasets/`. Check its size against your quota
first (`du -sh`, `quota -s` if available). Kaggle API tokens (`~/.kaggle/kaggle.json`)
must never be committed. `.gitignore` excludes `kaggle.json`.

## Annotation format

- **YOLO format**: one `.txt` per image with the same stem, one object per line:
  `<class_id> <x_center> <y_center> <width> <height>`, normalised to [0, 1].
- **Class ids**: `0 = smoke`, `1 = fire`. This matches `names: ['smoke', 'fire']`
  in the authors' YOLOv5 config (`data/dfire.yaml` in
  <https://github.com/pedbrgs/Fire-Detection>) and `configs/baseline.yaml`.
  Confirmed on the OneDrive train/val/test download (2026-10-13): class 1 has
  14,692 boxes and class 0 has 11,865, which matches upstream's fire and smoke
  totals exactly. A Kaggle re-upload could in principle re-map ids, so run the
  validator on any other copy.
- **Known label quirks** (found by `scripts/prepare_data.py`, left unmodified): 19
  zero-area boxes (width or height is 0) across all splits, and 7 boxes in `test`
  whose width or height slightly exceeds 1.0 (up to 1.056). The trainer should
  drop degenerate boxes and clip coordinates to [0, 1] when loading.
- **Empty label file** = background image with no fire or smoke. Upstream reports a
  large share of such images, which is intended and not an error.
- Upstream notes that some labelling tools produced negative coordinates. The
  validator counts these as `non_normalised_boxes` (a warning).

## Expected layout

```
<data_root>/
  train/images/*.jpg   train/labels/*.txt
  val/images/*.jpg     val/labels/*.txt      # may be named "valid" in some downloads
  test/images/*.jpg    test/labels/*.txt
```

Split directory names are configured in `configs/baseline.yaml` under
`dataset.splits`. Adjust them to match your download rather than renaming the
data. Only `train` is mandatory. Missing splits are reported and skipped.

## Validate the download

```bash
python scripts/prepare_data.py --config configs/baseline.yaml --data-root ~/datasets/dfire \
    --report ~/firewatch_runs/dfire_report.json \
    --write-data-yaml ~/firewatch_runs/dfire_data.yaml
```

The script never modifies the dataset and does not train anything. It reports,
per split, the image and label counts, background images, and boxes and images per
class. It also classifies problems:

- **Errors**: malformed or non-UTF-8 label files, class ids outside
  `class_names`, and a split with zero images.
- **Warnings**: images without a label file, labels without an image, duplicate
  image stems (e.g. `a.jpg` + `a.png`), and boxes outside [0, 1]. Negative
  coordinates fall into this last category.

The exit code is 0 when clean, 1 on errors (or on warnings with `--strict`), and 2
on config or path problems, including a missing `train` split. The JSON report is
written whenever the scan ran (exit 0 or 1). It includes `errors`, `warnings`,
`exit_code`, and per-split details with up to 20 example paths per problem type.
`--write-data-yaml` emits an Ultralytics/YOLOv5-style data file for whichever
splits were found.

The dataset root can also be supplied as `dataset.root` in the config or as the
`FIREWATCH_DATA_ROOT` environment variable (CLI > config > env var).

## Licence and citation

D-Fire is released under CC0-1.0. If you publish results, cite the dataset paper
listed in the upstream README (de Venâncio et al., *Neural Computing and
Applications*, 2022). If you use the surveillance videos, also cite the 2023 paper.
