import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class FridgeDataset(Dataset):
    def __init__(self, csv_file, img_dir,  transform=None):
        self.labels_data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.label_map = {"full": 0, "half-full": 1, "empty": 2}

        total = len(self.labels_data_frame)

        self.labels_data_frame = self.labels_data_frame[self.labels_data_frame.iloc[:, 8].apply(
            lambda x: os.path.exists(os.path.join(self.img_dir, x))
        )].reset_index(drop=True)

        missing = total - len(self.labels_data_frame)
        if missing > 0:
            print(f"Skipped {missing} missing images out of {total}.")
    
    def __len__(self):
        return len(self.labels_data_frame)
    
    def __getitem__(self, idx):

        # get the image filename from the csv (data frame)
        img_name = os.path.join(self.img_dir, self.labels_data_frame.iloc[idx, 8])
        image = Image.open(img_name).convert("RGB")
        
        # get the classigication label from the csv (data frame)
        label_str = self.labels_data_frame.iloc[idx, 2]
        label = self.label_map[label_str]

        # apply the transformation if any
        if self.transform:
            image = self.transform(image)


        return image, label
     
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # resize all images
    transforms.ToTensor(),           # convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])  # normalize like ImageNet
])

dataset = FridgeDataset(csv_file='data/corrected_annotations.csv', img_dir='data/images', transform=transform)
train_size = int(0.8*len(dataset))
validate_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validate_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

