# PI Generator

## Overview

The goal of this project is to demonstrate meaningful learning and ensure the generated points follow the same distribution as the dataset.

This repository includes two distinct approaches for achieving this objective:
- **MaskedTransformer-Based Approach**: A transformer-based architecture that efficiently handles high-dimensional sequential data.
- **Normalizing Flow (RealNVP)-Based Approach**: A flow-based generative model designed for exact likelihood estimation and precise data generation.

---

## Directory Structure

- **`flow_based/`**: Contains the source code for the Normalizing Flow (RealNVP) approach.
- **`mask_based/`**: Contains the source code for the MaskedTransformer approach.
- **`model/`**: Directory for trained model files.
- **`plot_flow/`**: Contains visualizations generated during the evaluation of the flow-based approach.
- **`plot_mask/`**: Contains visualizations generated during the evaluation of the MaskedTransformer approach.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ajishpradeep/pi_generator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pi_generator
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the main script to train or evaluate the MaskedTransformer model:
   ```bash
   python mask_based/pi_generator.py
   ```
   Similarly, run the following script to train or evaluate the flow-based approach using RealNVP:
   ```bash
   python flow_based/pi_generator_flow.py
   ```
2. Both **training** and **evaluation** functions are called by default. Since trained models are already provided in the `model/` directory, simply comment out the training function to run only the evaluation step.
3. View visualizations and results in the `plot_mask/` and `plot_flow/` directories, respectively.

---

## Problem Explanation

The goal is to train a model capable of learning the data distribution and generating similar points for the 5-dimensional input data `(x, y, r, g, b)`.

---

## Model Architecture

### 1. **MaskedTransformer-Based Approach**
The MaskedTransformer is a transformer-based neural network designed to handle high-dimensional sequential data effectively.

- Although flow-based models or Variational Autoencoders (VAEs) are sufficient for this task, the Transformer architecture was chosen to demonstrate a deeper understanding of its core principles and applicability. Its unique features provide additional advantages for addressing this task efficiently.
- The MaskedTransformer was selected for its ability to handle the unique challenges of sparse, high-dimensional data, such as `(x, y, r, g, b)` input points. The model embeds the input into a higher-dimensional space (`embed_dim=128`) using a linear layer, enabling the representation of spatial and color relationships. A learned positional encoding is added, providing spatial context to the embeddings, which is crucial for sparse datasets with irregular patterns. 
- The Transformer Encoder, comprising four layers (`num_layers=4`) and four attention heads (`num_heads=4`), leverages multi-head attention to capture complex dependencies between spatial and color features. The masking mechanism ensures the model focuses on relevant sequence elements, reducing noise and improving training efficiency.
- The output layers are modular, with dedicated linear layers for each dimension (`x, y, r, g, b`). Sigmoid activation is applied to color values (`r, g, b`) to constrain outputs to the valid range `[0, 1]`. The loss function is carefully designed to weight spatial (`x, y`) and color features (`r, g, b`) differently, emphasizing color accuracy.

---

### 2. **Normalizing Flow (RealNVP)-Based Approach**
The RealNVP-based approach was chosen for its ability to perform **exact likelihood estimation** and **invertible transformations**, enabling precise modeling of complex data distributions.

- Unlike VAEs, which approximate likelihoods, RealNVP computes exact likelihoods using the change-of-variables formula, making it ideal for tasks requiring explicit density modeling. The architecture consists of six coupling layers (`num_layers=6`) with alternating masking patterns, incrementally transforming input data into a structured latent space while ensuring computational stability.
- Each coupling layer employs two MLPs to learn **scale** and **translation** transformations for the unmasked portions of the input, capturing intricate nonlinear dependencies. A standard Gaussian distribution serves as the base distribution for sampling, providing a simple latent space and guaranteeing alignment with the true data distribution. This model’s explicit density estimation, invertibility, and ability to handle high-dimensional data efficiently make it a powerful tool for learning and generating data that closely follows the original distribution.

---

## Evaluation Results

Both models were evaluated using the following metrics:

- **Similarity Metrics:**
  - **Maximum Mean Discrepancy (MMD):** Captures global similarity and ensures the generated data follows the overall distribution.
  - **KL Divergence:** Provides detailed insights into the alignment of individual dimensions, focusing on specific features like spatial accuracy.
  - **Wasserstein Distance:** Evaluates geometric alignment, highlighting how well the model minimizes positional and color discrepancies.

### Results

#### MaskedTransformer Approach:
- **MMD:** 0.0019
- **KL Divergence:** x=0.0858, y=0.0534
- **Wasserstein Distances:**
  - x: 0.008623, y: 0.003177, r: 0.006073, g: 0.004483, b: 0.011655

#### Normalizing Flow (RealNVP) Approach:
- **MMD:** 0.0011
- **KL Divergence:** x=0.0535, y=0.0466
- **Wasserstein Distances:**
  - x: 0.010877, y: 0.006068, r: 0.011408, g: 0.010864, b: 0.006135

---

## Visual Analysis

Both approaches were further evaluated using:
- **Histograms**: Show alignment of real and generated data distributions.
- **Scatter Plots**: Highlight relationships between spatial and color features.
- **PCA Visualization**: Confirms similarity in reduced-dimensional representations.

Check the `plot_mask/` and `plot_flow/` directories for visualizations.
