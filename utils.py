import torch

def collate_fn(batch):
    """
    Custom collate function to pad audio spectrograms to the maximum width in the batch.
    """
    videos = torch.stack([b['video'] for b in batch])  # Stack video tensors
    labels = torch.tensor([b['label'] for b in batch])  # Stack labels
    ids = [b['id'] for b in batch]  # Collect IDs

    # Find the max width of spectrograms in the batch
    max_width = max(b['audio'].shape[1] for b in batch)
    padded_audios = []
    for b in batch:
        audio = b['audio']
        pad_width = max_width - audio.shape[1]
        padded_audio = torch.nn.functional.pad(audio, (0, pad_width))
        padded_audios.append(padded_audio)
    audios = torch.stack(padded_audios)

    return {"video": videos, "audio": audios, "label": labels, "id": ids}