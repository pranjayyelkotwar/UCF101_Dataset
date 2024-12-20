# UCF101 Dataset Preparation and Usage

This repository provides scripts and instructions to prepare and use the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) for multimodal machine learning tasks. Below are the steps to organize, preprocess, and load the dataset efficiently.

---

## Download the Dataset

1. Download the dataset from the [UCF101 official page](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar).  
2. Extract the downloaded files to your desired directory.

---

## Step 1: Organize the Dataset

The original dataset is split into `train`, `test`, and `val` directories, which can be inconvenient for certain workflows. To consolidate the data:

1. Run the script `move_data.py`:
   - Update the `base_path` variable to point to the extracted dataset directory.
   - Set the `output_path` variable to the desired output directory for consolidated data.

### Expected Directory Structure After Running `move_data.py`:

```
dataset/
  ├── Class1/
  │     ├── video1.avi
  │     ├── video2.avi
  ├── Class2/
  │     ├── video1.avi
  │     ├── video2.avi
  ...
```

---


## Step 2: Preprocess the Videos and Audio

To make data loading easier, videos are converted to `.mp4` format, and audio is extracted as `.wav`. 

Run the `convert_video.py` script:
- Update the following variables in the script:
  - `input_dir`: Path to the folder containing videos (e.g., `"dataset"`).
  - `output_video_dir`: Path for storing the converted `.mp4` videos.
  - `output_audio_dir`: Path for storing extracted `.wav` audio files.

### Directory Structure After Conversion:

```
converted_videos/
  ├── Class1/
  │     ├── video1.mp4
  │     ├── video2.mp4
  ├── Class2/
  │     ├── video1.mp4
  │     ├── video2.mp4
  ...

extracted_audio/
  ├── Class1/
  │     ├── video1.wav
  │     ├── video2.wav
  ├── Class2/
  │     ├── video1.wav
  │     ├── video2.wav
  ...
```

---

manually put both these directories into one and rename to just video and audio and continue

## Step 3: Load the Dataset with PyTorch

To simplify data loading for machine learning tasks, use the `UCF101MultimodalDataset` class provided in `dataset_class.py`. 

### Features:
- Supports loading both video and audio modalities.
- Samples a fixed number of frames from each video.
- Converts audio to spectrograms and normalizes them.

### Example Usage:

```python
from dataset_class import UCF101MultimodalDataset

# Initialize the dataset
dataset = UCF101MultimodalDataset(root_dir="final_dataset")

# Fetch a sample
sample = dataset[3]
video, audio, label, id = sample["video"], sample["audio"], sample["label"], sample["id"]

# Print shapes and metadata
print("Video shape:", video.shape)  # e.g., torch.Size([8, 3, 224, 224])
print("Audio shape:", audio.shape)  # e.g., torch.Size([128, 346])
print("Label:", label)              # e.g., 0
print("ID:", id)                    # e.g., v_BalanceBeam_g23_c03
```

---

## Additional Notes

1. **Dependencies:** Ensure the following Python libraries are installed:
   - `torch`, `torchvision`, `torchaudio`
   - `moviepy`, `tqdm`
2. **Customization:** 
   - Modify `num_frames_per_clip` in the dataset class to change the number of frames sampled per video.
   - Update transformations for both video and audio as needed for your model.

---