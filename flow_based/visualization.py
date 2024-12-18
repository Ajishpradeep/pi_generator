import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance, entropy
import seaborn as sns
import cv2
from sklearn.decomposition import PCA
from scipy.stats import entropy, wasserstein_distance
import os

IMG_SIZE = 300

def save_points_as_image(points, out_path="plot_flow/gen_points.png", img_size=IMG_SIZE):
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for x_norm, y_norm, r, g, b in points:
        x_pix = int(np.clip(x_norm * (img_size - 1), 0, img_size - 1))
        y_pix = int(np.clip(y_norm * (img_size - 1), 0, img_size - 1))
        canvas[y_pix, x_pix] = (np.clip([b, g, r], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(out_path, canvas)

def save_and_close(fig, filename):
    output_dir = "plot_flow"
    os.makedirs(output_dir, exist_ok=True)  
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def plot_dimwise_histograms(real_data, gen_data, dim_labels=['x', 'y', 'r', 'g', 'b']):
    fig, axes = plt.subplots(1, len(dim_labels), figsize=(15, 3))
    for i, ax in enumerate(axes):
        sns.histplot(real_data[:, i], color='blue', stat='density', alpha=0.5, kde=True, ax=ax, label="Real")
        sns.histplot(gen_data[:, i], color='red', stat='density', alpha=0.5, kde=True, ax=ax, label="Gen")
        ax.set_title(dim_labels[i])
    axes[0].legend()
    save_and_close(fig, "histogram.png")

def plot_pairwise_scatter(real_data, gen_data, indices=(0, 1), dim_labels=['x', 'y']):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for data, ax, title in zip([real_data, gen_data], axes, ["Real", "Generated"]):
        colors = np.clip(data[:, 2:5], 0, 1)
        ax.scatter(data[:, indices[0]], data[:, indices[1]], c=colors, s=5)
        ax.set_title(f"{title}: {dim_labels[0]} vs {dim_labels[1]}")
        ax.invert_yaxis()
    save_and_close(fig, "scatter.png")

def pca_scatter_plot(real_data, gen_data):
    pca = PCA(n_components=2).fit(np.vstack((real_data, gen_data)))
    real_pca, gen_pca = pca.transform(real_data), pca.transform(gen_data)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', s=5, alpha=0.5, label='Real')
    ax.scatter(gen_pca[:, 0], gen_pca[:, 1], c='red', s=5, alpha=0.5, label='Generated')
    ax.legend()
    ax.set_title("PCA Projection")
    save_and_close(fig, "pca_scatter.png")

def compute_mmd(x, y, sigma=0.2):
    x_sq = np.sum(x**2, axis=1).reshape(-1, 1)
    y_sq = np.sum(y**2, axis=1).reshape(-1, 1)
    xx = np.exp(- (x_sq - 2 * np.dot(x, x.T) + x_sq.T) / (2 * sigma**2))
    yy = np.exp(- (y_sq - 2 * np.dot(y, y.T) + y_sq.T) / (2 * sigma**2))
    xy = np.exp(- (x_sq - 2 * np.dot(x, y.T) + y_sq.T) / (2 * sigma**2))
    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_kl_divergence(real_data, gen_data, bins=100):
    hist_real, bin_edges = np.histogram(real_data, bins=bins, density=True)
    hist_gen, _ = np.histogram(gen_data, bins=bin_edges, density=True)
    hist_real += 1e-10
    hist_gen += 1e-10
    hist_real /= hist_real.sum()
    hist_gen /= hist_gen.sum()
    return entropy(hist_real, hist_gen)

def compute_wasserstein(real_data, gen_data):
    distances = {}
    for i, label in enumerate(['x', 'y', 'r', 'g', 'b']):
        distances[label] = wasserstein_distance(real_data[:, i], gen_data[:, i])
    return distances