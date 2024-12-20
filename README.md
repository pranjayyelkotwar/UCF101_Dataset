# **UCF101 Dataset Preparation and Usage**

This repository provides scripts and guidelines to organize, preprocess, and load the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) for multimodal machine learning tasks, such as video and audio analysis.

---

## **Download the Dataset**

1. Download the dataset from the [UCF101 official page](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar).  
2. Extract the `.rar` file into your preferred directory.

---

## **Step 1: Organize the Dataset**

The dataset is initially divided into `train`, `test`, and `val` splits, which might not be ideal for certain workflows. To merge all videos into a single directory:

1. **Run `move_data.py`:**
   - Update the `base_path` variable to the path of the extracted dataset.
   - Set the `output_path` variable to your desired output directory.

After running the script, the directory structure should look like this:

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

## **Step 2: Preprocess Videos and Audio**

To simplify dataset loading, videos are converted to `.mp4`, and audio is extracted as `.wav` files. Follow these steps:

1. **Run `convert_video.py`:**
   - Update the following variables in the script:
     - `input_dir`: Path to the dataset folder (e.g., `"dataset"`).
     - `output_video_dir`: Directory for converted `.mp4` videos.
     - `output_audio_dir`: Directory for extracted `.wav` audio files.

2. After running the script, you will have the following directory structure:

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

3. **Manually merge the directories:**
   - Combine `converted_videos` and `extracted_audio` into a single directory:
     ```
     final_dataset/
       ├── video/
       ├── audio/
     ```

---

## **Step 3: Load the Dataset with PyTorch**

The `UCF101MultimodalDataset` class in `dataset_class.py` enables efficient data loading for video and audio inputs.

### **Features:**
- Samples a fixed number of frames per video.
- Converts audio to spectrograms and normalizes them.
- Returns video frames, audio spectrograms, labels, and video IDs.

### **Example Usage:**

```python
from dataset_class import UCF101MultimodalDataset

# Initialize the dataset
dataset = UCF101MultimodalDataset(root_dir="final_dataset")

# Fetch a sample
sample = dataset[3]
video, audio, label, id = sample["video"], sample["audio"], sample["label"], sample["id"]

# Print shapes and metadata
print("Video shape:", video.shape)  # Example: torch.Size([8, 3, 224, 224])
print("Audio shape:", audio.shape)  # Example: torch.Size([128, 346])
print("Label:", label)              # Example: 0
print("ID:", id)                    # Example: v_BalanceBeam_g23_c03
```

---

## **Step 4: Handle Variable Audio Sizes**

Due to variable audio sizes, standard PyTorch dataloaders may encounter issues. To address this, use the custom `collate_fn` provided in `utils.py` when creating the dataloader.

---

## **Summary**

This repository streamlines the preparation of the UCF101 dataset for multimodal tasks. Follow the steps to:
1. Consolidate the dataset structure.
2. Preprocess video and audio files.
3. Efficiently load data using PyTorch.

With these tools, you can easily integrate UCF101 into your machine learning pipelines. 