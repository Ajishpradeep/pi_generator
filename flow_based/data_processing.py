import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class ColoredPIDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_dataset_from_npy(image_path, xs_path, ys_path):
    xs = np.load(xs_path).astype(np.int32)
    ys = np.load(ys_path).astype(np.int32)
    image = np.array(Image.open(image_path), dtype=np.float32) / 255.0

    h, w, _ = image.shape
    x_coords = xs / w
    y_coords = ys / h
    rgb_values = image[xs, ys]

    return np.column_stack((x_coords, y_coords, rgb_values)).astype(np.float32)
