# YOLO Training GUI Application

This project provides a user-friendly Graphical User Interface (GUI) built with PyQt5 for training custom YOLO models using the `ultralytics` library. It simplifies the process of dataset validation, configuration, training, and model exporting.

## Features

* **Intuitive GUI**: Easily configure and run training without command-line arguments.
* **Configurable Training Profiles**: Select from pre-defined profiles (`default`, `large_dataset`, `focus_accuracy`, etc.) or use default settings.
* **Automated Dataset Handling**:

  * Validates image and label files, cleaning corrupted or invalid data.
  * Automatically splits the dataset into training, validation, and test sets.
  * Supports symbolic links to save disk space.
  * Converts JSON labels to the required YOLO `.txt` format.
* **Training Options**: Toggle data augmentation, mixed-precision (FP16) training, and resuming from a checkpoint.
* **Real-time Logging**: View training progress and logs directly in the application window.
* **Multi-precision Model Saving**: Automatically saves the best model in `FP32` (`best.pt`), `FP16` (`best_fp16.pt`), and `TorchScript` formats for flexible deployment.

## ⚙️ Environment Setup

This section provides detailed instructions to set up the environment correctly. Following these steps is crucial to avoid common issues like Qt plugin conflicts.

### Prerequisites

* **Git**: To clone the repository.
* **Conda / Miniforge**: It is **highly recommended** to use a Conda environment to manage dependencies and avoid conflicts. Miniforge is a lightweight alternative to Anaconda.
* **(Optional but Recommended) NVIDIA GPU with CUDA**: For significantly faster training. Ensure your NVIDIA drivers are installed.

### Step-by-Step Installation



1. **Create and Activate a Conda Environment**

   We will use Python 3.10, as it is a stable version compatible with the required libraries.

   ```bash
   # Create a new environment named 'yolo_train_env'
   conda create --name yolo_train_env python=3.10 -y

   # Activate the environment
   conda activate yolo_train_env
   ```

2. **Create a `requirements.txt` file**

   Create a new file named `requirements.txt` in the project root and add the following content:

   ```txt
   # Core deep learning and computer vision
   torch
   torchvision
   torchaudio
   ultralytics
   opencv-python-headless
   scikit-learn

   # GUI Framework
   PyQt5
   ```

3. **Install Dependencies**

   Now, install all the required packages from the `requirements.txt` file. We have two paths: GPU (recommended) or CPU-only.

   #### For GPU Users (Recommended)

   First, install PyTorch with CUDA support. Go to the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the correct command for your specific CUDA version. For example, for CUDA 12.1:

   ```bash
   # IMPORTANT: Use the command from the PyTorch website for your system!
   # The command below is an EXAMPLE for pip with CUDA 12.1.
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   After PyTorch is installed, install the rest of the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   #### For CPU-Only Users

   If you don't have an NVIDIA GPU, you can install the CPU version of PyTorch and the other packages directly:

   ```bash
   pip install -r requirements.txt
   ```

   > **💡 Important Note on `opencv-python-headless`**:
   > We explicitly use `opencv-python-headless` instead of `opencv-python`. This is because the standard `opencv-python` package bundles its own Qt libraries, which conflict with `PyQt5` and cause the application to crash with an `xcb` plugin error. The `headless` version contains all the necessary image processing functions (`cv2.imread`, etc.) without the conflicting GUI components.

## 📁 Project Structure

```
.
├── main_gui.py         # Main application file with the PyQt5 GUI
├── training_core.py    # Core logic for the YOLO training pipeline (runs in a separate thread)
├── data_utils.py       # Functions for dataset validation, preparation, and label conversion
├── config.py           # All configurations: model selection, training profiles, category mappings
├── requirements.txt    # List of Python dependencies
├── label/              # (Example) Default directory for your raw image and JSON label files
└── README.md           # This file
```

After running a training session, new directories will be created:

* `runs/train/`: Contains the output from the `ultralytics` training, including saved models, logs, and validation results.
* `train/`, `val/`, `test/`: Contain the prepared dataset (images and labels), possibly as symbolic links.
* `data.yaml`: The data configuration file automatically generated for YOLO.

## 📖 How to Use

### 1. Prepare Your Dataset

* Place your images (e.g., `.jpg`, `.png`) and their corresponding JSON annotation files in a single folder. The default location is a folder named `label` in the project root.
* Each image file (e.g., `image1.jpg`) must have a matching label file with the same base name (e.g., `image1.json`).
* The JSON file should have a structure like this:

  ```json
  {
    "labels": [
      {
        "name": "bottle",
        "x1": 150, "y1": 200,
        "x2": 250, "y2": 400
      },
      {
        "name": "can",
        "x1": 300, "y1": 250,
        "x2": 380, "y2": 350
      }
    ]
  }
  ```
* Make sure the class names in your JSON (`"name": "bottle"`) are defined in the `CATEGORY_MAPPING` dictionary in `config.py`.

### 2. Run the Application

Ensure your conda environment is activated, then run the main GUI script:

```bash
python main_gui.py
```

### 3. Configure and Start Training

1. **Select Data Folder**: If your dataset is not in the default `./label` folder, click "Select Data Folder" to choose the correct location.
2. **Choose a Training Profile**: Select a profile from the dropdown menu that best suits your dataset and goals.
3. **Set Options**:

   * Check "Use Data Augmentation" to improve model robustness (recommended).
   * Check "Use Mixed Precision" for faster training on modern GPUs (recommended for NVIDIA RTX series or newer).
   * Check "Resume Training" if you want to continue a previously stopped training session.
4. **Start Training**: Click the "Start Training" button. The progress will be displayed in the log area.
5. **Monitor**: You can monitor the training process in the log window. The "Stop Training" button can be used to interrupt the process gracefully.
6. **Results**: Once training is complete, the final models will be saved in the latest directory inside `runs/train/weights/`.

## ⚠️ Troubleshooting

**Error: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb" ...`**

This is the most common error and is caused by a conflict between the Qt libraries bundled with `opencv-python` and those used by `PyQt5`.

**Solution**: Ensure you have installed `opencv-python-headless` as specified in the installation instructions.

1. Deactivate and remove the old environment if necessary.
2. Follow the **Environment Setup** instructions carefully, paying special attention to the `requirements.txt` file and the note about `opencv-python-headless`.
