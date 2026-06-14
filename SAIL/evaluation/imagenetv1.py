from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

class ImageNetWithPaths(ImageNet):
    def __getitem__(self, index):
        # Get the original image and target from the ImageNet dataset
        image, target = super().__getitem__(index)
        
        # Get the image name
        image_name = self.samples[index][0].split("/")[-1].split(".")[0]
        
        # Return image, target, and image path
        return image, target, image_name