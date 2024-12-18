import torch
import os
from visualization import *


def logit_transform(x, lam=1e-6):
    x = lam + (1 - 2 * lam) * x
    return torch.log(x) - torch.log(1 - x)

def inv_logit_transform(u, lam=1e-6):
    return (torch.sigmoid(u) - lam) / (1 - 2 * lam)

def train_flow(flow, dataloader, device, epochs=50, lr=1e-3):
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True) 
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    flow.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            u = logit_transform(batch)
            optimizer.zero_grad()
            log_prob = flow.log_prob(u)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        torch.save(flow.state_dict(), os.path.join(model_dir, "best_flow.pth"))

def evaluate_flow(flow, real_data, device):
    flow.load_state_dict(torch.load(os.path.join("model", "best_flow.pth"), map_location=device))
    flow.eval()

    with torch.no_grad():
        z = torch.randn(len(real_data), 5, device=device)
        gen_u = flow.inverse(z)
        gen_x = inv_logit_transform(gen_u)
        gen_x = torch.clamp(gen_x, 0.0, 1.0)
    gen_data = gen_x.cpu().numpy()

    plot_dimwise_histograms(real_data, gen_data, dim_labels=['x', 'y', 'r', 'g', 'b'])
    plot_pairwise_scatter(real_data, gen_data, indices=(0, 1), dim_labels=['x', 'y'])
    pca_scatter_plot(real_data, gen_data)
    save_points_as_image(gen_data, img_size=300)

    mmd_value = compute_mmd(real_data, gen_data, sigma=0.1)
    print(f"MMD: {mmd_value:.4f}")
    kl_x = compute_kl_divergence(real_data[:, 0], gen_data[:, 0])
    kl_y = compute_kl_divergence(real_data[:, 1], gen_data[:, 1])
    print(f"KL Divergence: x={kl_x:.4f}, y={kl_y:.4f}")
    wasserstein = compute_wasserstein(real_data, gen_data)
    print("Wasserstein Distances:")
    for dim, distance in wasserstein.items():
        print(f"  {dim}: {distance:.6f}")
