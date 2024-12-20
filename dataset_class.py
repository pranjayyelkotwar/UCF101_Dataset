import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tv
import torchaudio
import torchaudio.transforms as Ta
from PIL import Image

class UCF101MultimodalDataset(Dataset):
    def __init__(self, root_dir, num_frames_per_clip=8, transform_video=None, transform_audio=None, spec_mean=0.0, spec_std=1.0):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            num_frames_per_clip (int): Number of frames to sample per video.
            transform_video (callable, optional): Transform for video frames.
            transform_audio (callable, optional): Transform for audio spectrograms.
            spec_mean (float): Mean for spectrogram normalization.
            spec_std (float): Std for spectrogram normalization.
        """
        self.sampling_frequency = 16000  # Fixed sampling frequency for audio

        self.root_dir = root_dir
        self.num_frames_per_clip = num_frames_per_clip
        self.transform_video = transform_video or self._default_video_transforms()
        self.transform_audio = transform_audio or self._default_audio_transforms(spec_mean, spec_std)

        self.data = self._load_data()

    def _load_data(self):
        """
        Parse the dataset directory structure.
        Returns a list of tuples (video_path, audio_path, label).
        """
        data = []
        video_files_dir = os.path.join(self.root_dir, "video")
        audio_files_dir = os.path.join(self.root_dir, "audio")
        class_dirs = os.listdir(video_files_dir)

        for class_idx, class_dir in enumerate(class_dirs):
            video_dir = os.path.join(video_files_dir, class_dir)
            audio_dir = os.path.join(audio_files_dir, class_dir)
            video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
            audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
            
            for vf, af in zip(video_files, audio_files):
                id = os.path.basename(vf).split('.')[0]
                data.append((vf, af, class_idx, id))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, audio_path, label, id = self.data[idx]

        # Load video frames
        video_frames = self._load_video_as_frames(video_path)
        video_frames = self.transform_video(video_frames)

        # Load audio spectrogram
        audio_spectrogram = self._load_audio(audio_path)
        audio_spectrogram = self.transform_audio(audio_spectrogram)

        return {"video": video_frames, "audio": audio_spectrogram, "label": label, "id": id}

    def _load_video_as_frames(self, video_path):
        """
        Load video as frames, sample a fixed number of frames, and stack them.
        """
        from torchvision.io import read_video
        video, _, _ = read_video(video_path)
        frame_count = video.shape[0]
        indices = np.linspace(0, frame_count - 1, self.num_frames_per_clip, dtype=int)
        sampled_frames = video[indices]
        return sampled_frames.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)

    def _load_audio(self, audio_path):
        """
        Load audio as waveform, process it into a spectrogram, and normalize.
        """
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        # Use mono audio (left channel if stereo)
        waveform = waveform[0]
        # Resample if necessary
        if sample_rate != self.sampling_frequency:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_frequency)
        # Normalize waveform
        waveform = (waveform - torch.mean(waveform)) / (torch.std(waveform) + 1e-6)
        return waveform

    def _default_video_transforms(self):
        """
        Default transformations for video frames.
        """
        return Tv.Compose([
            Tv.ConvertImageDtype(torch.float32),
            Tv.Resize((224, 224)),  # Resize frames to 224x224
            Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ])

    def _default_audio_transforms(self, spec_mean, spec_std):
        """
        Default transformations for audio spectrograms.
        """
        mel = Ta.MelSpectrogram(
            sample_rate=self.sampling_frequency,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=128,
        )
        a2d = Ta.AmplitudeToDB()

        # return lambda waveform: (a2d(mel(waveform)) - spec_mean) / spec_std
        
        def transform(waveform):
            return (a2d(mel(waveform)) - spec_mean) / spec_std
        
        return transform

    

# Example usage
# dataset = UCF101MultimodalDataset(
#     root_dir="final_dataset"
# )
# video , audio , label , id = dataset[3]['video'] , dataset[3]['audio'] , dataset[3]['label'] , dataset[3]['id']
# print("Video shape:", video.shape)
# print("Audio shape:", audio.shape)
# print("Label:", label)
# print("ID:", id)
# # Video shape: torch.Size([8, 3, 224, 224])
# # Audio shape: torch.Size([128, 346])
# # Label: 0
# # ID: v_BalanceBeam_g23_c03