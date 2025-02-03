import os
from os.path import join
from torch.utils.data import Dataset
from PIL import Image

class FabricDataset(Dataset):
    def __init__(self, data_path, transform=False):
        super().__init__()
        self.label_path = join(data_path, 'label')
        self.image_path = join(data_path, 'image')
        self.image_filenames = [x for x in os.listdir(self.image_path) if '.png' in x]
        self.transform = transform
        
    def __getitem__(self, index):
        label = Image.open(join(self.label_path, self.image_filenames[index])).convert('RGB')
        image = Image.open(join(self.image_path, self.image_filenames[index])).convert('RGB')
        
        if self.transform:
            label = self.transform(label)
            image = self.transform(image)
        return image, label
            
    def __len__(self):
        return len(self.image_filenames)