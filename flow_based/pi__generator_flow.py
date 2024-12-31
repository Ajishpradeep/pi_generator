import torch
from data_processing import create_dataset_from_npy, ColoredPIDataset
from torch.utils.data import DataLoader
from normalizing_flow_model import RealNVP
from normalizing_flow_training import train_flow, evaluate_flow

if __name__ == "__main__":
    IMG_SIZE = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_data = create_dataset_from_npy(
        "sparse_pi_colored.jpg", "pi_xs.npy", "pi_ys.npy"
    )
    dataset = ColoredPIDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    flow = RealNVP(dim=5, num_layers=6, hidden_dim=256).to(device)
    train_flow(flow, dataloader, device, epochs=50, lr=1e-3)
    print("==========Evaluating the model===========")
    evaluate_flow(flow, real_data, device=device)
