import torch
from torch.utils.data import Dataset
from skimage import io
from main import extract_dataset_file


class TrafficLightDataset(Dataset):
    def __init__(self, cropped_image_file, root_dir, transform=None):
        self.annotations = extract_dataset_file(cropped_image_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image = io.imread(self.annotations[index, 0])
        if self.annotations[index, 2]:
            y_label = torch.tensor(1)
        else:
            y_label = torch.tensor(0)
        if self.transform:
            image = self.transform(image)
        return image, y_label
