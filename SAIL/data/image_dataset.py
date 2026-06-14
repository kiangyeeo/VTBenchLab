from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os
from PIL import ImageFile
from .utils import load_data
from transformers.image_processing_base import BatchFeature

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], image_processor, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                    return img  # Return directly if custom transform is used
                else:
                    # Return the dictionary from image_processor
                    return self.image_processor(img, return_tensors="pt")
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            # Return a placeholder with the same format as image_processor output
            return {'pixel_values': torch.zeros((1, 3, 224, 224))}

def collate_fn(batch):
    # Stack all tensors in the batch along dim=0 to create a batch of shape [n,3,224,224]
    tensors = [item['pixel_values'] for item in batch]
    tensors = [t.squeeze(0) if t.dim() == 4 else t for t in tensors] # Remove batch dim if present

    return BatchFeature(data={'pixel_values': torch.stack(tensors, dim=0)}, tensor_type="pt")

def create_image_dataloader(
    image_paths: List[str],
    image_processor,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    transform = None
) -> DataLoader:
    dataset = ImageDataset(image_paths, image_processor, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    ) 

if __name__ == "__main__":
    from .data_config import DATADIR
    from transformers import AutoImageProcessor
    sentences, image_paths = load_data(DATADIR['dreamclipcc3m'], 'train', 'image')
    image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    dataloader = create_image_dataloader(image_paths, image_processor, batch_size=128, num_workers=2, shuffle=False)
    breakpoint()
    for batch in dataloader:
        print(batch.shape)