# YOLO Trainer (CLI + GUI)

Train Ultralytics YOLO models from a simple **CLI** or a friendly **PyQt6 GUI**.  
Includes dataset checking, JSON→YOLO label conversion, training orchestration, and model export (FP16 + TorchScript).  
Tested with the Ultralytics YOLO family (e.g., `yolov8*`, `yolo11*`, `yolo12*`).   

---

## 1) Quick start

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate                                # Windows

# 2) Install dependencies
pip install -r requirements.txt

# (If you use GPU) Install the right Torch build for your CUDA/OS
# See: https://pytorch.org/get-started/locally/

# 3) Put your dataset under ./label (see format below)

# 4a) Run the CLI
python train_cli.py

# 4b) Or launch the GUI
python train_gui.py
````

---

## 2) Dataset format

Place **images** and matching **JSON** files together in `./label/`.

```
label/
  IMG_0001.jpg
  IMG_0001.json
  IMG_0002.png
  IMG_0002.json
  ...
```

Each JSON must contain a top-level `"labels"` array. Every label entry should include the bounding box coordinates **x1, y1, x2, y2** (pixels) and a class `"name"`.
The pipeline validates images and JSON, converts boxes to YOLO format, and writes split folders (`train/`, `val/`, `test/`) with `images/` and `labels/`. At least **15** valid image–label pairs are required.

### Class mapping (built-in)

| Class name in JSON                              | ID |
| ----------------------------------------------- | -: |
| Kitchen\_waste, potato, daikon, carrot          |  0 |
| Recyclable\_waste, bottle, can                  |  1 |
| Hazardous\_waste, battery, drug, inner\_packing |  2 |
| Other\_waste, tile, stone, brick                |  3 |

The mapping above is applied automatically during conversion. Unknown class names are skipped with a warning.

---

## 3) Using the CLI

```bash
# Defaults
python train_cli.py

# With config file
python train_cli.py --config config.yaml

# Use a preset
python train_cli.py --preset servermode

# Override common params
python train_cli.py --model yolo11s.pt --epochs 200 --batch-size 16 --imgsz 640 --device 0

# Resume training
python train_cli.py --resume

# Generate a template config
python train_cli.py --generate-config my_config.yaml
```

Key options:

* **Model & data**: `--model`, `--data-path`, `--train-split/--val-split/--test-split`, `--no-symlinks`
* **Training**: `--epochs`, `--batch-size`, `--imgsz`, `--device`, `--workers`, `--resume`
* **Optimization**: `--optimizer {Adam,AdamW,SGD,RMSProp}`, `--lr0`, `--lrf`, `--momentum`, `--weight-decay`, `--patience`
* **Augmentation**: `--augment`, `--degrees`, `--scale`, `--fliplr`, `--flipud`
* **Advanced**: `--mixed-precision/--fp16`, `--multi-scale`, `--rect`, `--cache`, `--dropout`, `--cos-lr`
* **Outputs**: `--project`, `--name`, `--exist-ok`
* **Utility**: `--verbose`, `--quiet`, `--dry-run`

Dataset splits **must sum to 1.0**, or the run will fail fast. A configuration snapshot is saved to `training_config.yaml` (or to `<project>/<name>/training_config.yaml` if `--project` is set).

**Presets** (`default`, `large_dataset`, `small_dataset`, `focus_accuracy`, `focus_speed`, `servermode`) tune batch size, epochs, LR schedule, etc. The `servermode` preset favors GPU/FP16, cosine LR, overlap masks, and multi-scale training.

---

## 4) Using the GUI

Launch:

```bash
python train_gui.py
```

The GUI mirrors the CLI options across tabs (Basic, Training, Optimization, Augmentation, Advanced) and supports:

* Loading/saving/exporting configs
* Live progress & logs
* Start/stop training
* Preset selection (updates fields automatically)

It enforces that dataset splits sum to 1.0 and that the dataset path exists before starting. A `training_config.yaml` is saved for reproducibility.

---

## 5) What the pipeline does

1. **Integrity check** – scans `data_path` for images (`.jpg/.jpeg/.png`) and validates matching JSON. Invalid or tiny images and malformed JSON are skipped.
2. **Prepare splits** – builds `train/`, `val/`, `test/` with `images/` and `labels/`, using symlinks (or copies) and converting JSON → YOLO `.txt` labels.
3. **Create `data.yaml`** – points YOLO to split folders and defines 4 classes.
4. **Train** – calls Ultralytics `YOLO(model).train(**args)` with your settings/preset. Mixed precision is enabled when requested and not on CPU.
5. **Export** – loads `weights/best.pt`, writes `best_fp16.pt`, and exports a **TorchScript** model.

---

## 6) Outputs

* Runs are saved under `<project>/<name>` (or the current directory if no `project` given).
* Look for `weights/best.pt`, `weights/best_fp16.pt`, and TorchScript export artifacts.
* A copy of the effective configuration is saved as `training_config.yaml`.

---

## 7) Tips & troubleshooting

* **“Dataset splits must sum to 1.0”** – Adjust `train/val/test` so they add up exactly to 1.0. The CLI/GUI both enforce this.
* **“No valid data pairs found”** – Ensure every image has a well-formed JSON (`labels` array, valid coordinates, known class names). You need **≥15** valid pairs.
* **CPU-only machines** – Batch size is capped and FP16 is disabled automatically.
* **GPU selection** – Use `--device 0` (or `1/2/3`) to pick a GPU; `auto` picks GPU if available.

---

## 8) Config file

You can generate and edit a YAML config:

```bash
python train_cli.py --generate-config my_config.yaml
# or load/save via the GUI
```

Every CLI/GUI setting maps to `TrainingConfig` fields and is serialized to YAML for reproducibility.

---
