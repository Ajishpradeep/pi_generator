import torch
import os
import numpy as np
from model import MaskedTransformer
from visualization import (
    plot_dimwise_histograms,
    plot_pairwise_scatter,
    pca_scatter_plot,
    save_points_as_image,
    compute_mmd,
    compute_kl_divergence,
    compute_wasserstein,
)

def train(model, dataloader, criterion, optimizer, device, epochs=10):
    best_loss = float("inf")
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            inputs = batch.unsqueeze(1).repeat(1, 5, 1)
            targets = [batch[:, i] for i in range(5)]
            outputs = model(inputs)
            losses = [criterion(outputs[i], targets[i].unsqueeze(-1)) for i in range(5)]
            loss = sum(
                [1.0 * losses[0], 1.0 * losses[1]] + [10.0 * l for l in losses[2:]]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "mask_model.pth"))
            # print("Best model saved.")


def evaluate_model(dataset, real_data, device):
    model = MaskedTransformer().to(device)
    model.load_state_dict(
        torch.load("model/mask_model.pth", map_location=device, weights_only=True)
    )
    model.eval()

    generated_points = []
    with torch.no_grad():
        for _ in range(len(dataset)):
            init_sample = dataset[np.random.randint(0, len(dataset))]
            input_point = init_sample.unsqueeze(0).unsqueeze(1).to(device)
            outputs = model(input_point.repeat(1, 5, 1))
            generated_points.append([o.item() for o in outputs])
    gen_data = np.clip(np.array(generated_points), 0.0, 1.0)

    # Evaluations
    plot_dimwise_histograms(real_data, gen_data, dim_labels=["x", "y", "r", "g", "b"])
    plot_pairwise_scatter(real_data, gen_data, indices=(0, 1), dim_labels=["x", "y"])
    pca_scatter_plot(real_data, gen_data)
    save_points_as_image(gen_data, out_path="gen_points.png", img_size=300)
    mmd_value = compute_mmd(real_data, gen_data, sigma=0.1)
    print(f"MMD: {mmd_value:.4f}")
    kl_x = compute_kl_divergence(real_data[:, 0], gen_data[:, 0])
    kl_y = compute_kl_divergence(real_data[:, 1], gen_data[:, 1])
    print(f"KL Divergence: x={kl_x:.4f}, y={kl_y:.4f}")
    wasserstein = compute_wasserstein(real_data, gen_data)
    print("Wasserstein Distances:")
    for dim, distance in wasserstein.items():
        print(f"  {dim}: {distance:.6f}")
