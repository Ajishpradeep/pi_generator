from data_processing import create_dataset_from_npy, ColoredPIDataset
from model import MaskedTransformer
from training import train, evaluate_model
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "sparse_pi_colored.jpg"
    xs_path = "pi_xs.npy"
    ys_path = "pi_ys.npy"
    
    real_data = create_dataset_from_npy(image_path, xs_path, ys_path)
    dataset = ColoredPIDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MaskedTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Training
    train(model, dataloader, criterion, optimizer, device, epochs=20)

    # Evaluation
    print("==========Evaluating the model===========")
    evaluate_model(dataset, real_data, device)

if __name__ == "__main__":
    main()
