# PI generator

## Overview
 The goal is to demonstrate meaningful learning and to ensure the generated points follow the same distribution as the dataset.

## Directory Structure

- **`src/`**: Contains the source code for the project.
- **`model/`**: Directory for the trained model.
- **`plot/`**: Contains visualizations generated during evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ajishpradeep/pi_generator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pi_generator
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script to train or evaluate the model:
   ```bash
   python src/pi_generator.py
   ```
2. Both training and Evaluation functions are called by default. Since the trained model is already in  `model/` directory, SImply comment the training call and execue the evalution if needed. 

3. View visualizations and results in the `plot/` directory.

## Problem Explanation

The model should learn the data distribution and generate similar points for the 5 dimentioanl input data.

## Model Architecture

The chosen model is a **MaskedTransformer**, a transformer-based neural network designed to handle high-dimensional sequential data. 

### Architecture Details

1. **Input Embedding:**
   - The input `[x, y, r, g, b]` is passed through a linear layer to transform it into a higher-dimensional embedding space (`embed_dim=128`).
   - This embedding allows the model to learn more abstract representations of the input data.

2. **Positional Encoding:**
   - Positional encoding is added to the embeddings to provide the model with information about the order of the input sequence.
   - A learned positional encoding is used, enhancing flexibility for sparse data.

3. **Transformer Encoder:**
   - A stack of 4 transformer encoder layers (`num_layers=4`) processes the embedded input.
   - Each layer uses 4 attention heads (`num_heads=4`) to focus on different relationships in the data.
   - Masking ensures that the model only considers relevant parts of the sequence during training.

4. **Output Layers:**
   - The output is generated using separate linear layers for each dimension (`x`, `y`, `r`, `g`, `b`).
   - For color dimensions (`r`, `g`, `b`), sigmoid activation ensures outputs are in the valid range [0, 1].

### Why This Architecture Was Chosen

   Though flow based or VAE are enogh for this task, The transformers architeure being choosen to demonstrate the core undertanding of the architure. Although the below mentioned functionalies of transformer do help addressing the task with efficiency in a way.

1. **Handling Sparse Data:**
   - The positional encoding and masking mechanisms make the transformer effective for irregular datasets like sparse images.

2. **Modularity:**
   - The architecture separates input encoding, positional information, and output generation, making it easy to adapt or extend.

3. **Focus on Relationships:**
   - Attention heads allow the model to understand complex interactions between position (`x, y`) and color (`r, g, b`).

## Evaluation Results

The model's performance was evaluated through:

- **Similarity Metrics:** 
  - **Maximum Mean Discrepancy (MMD):** Measures how close the generated points are to the real ones.
  - **KL Divergence:** Compares distributions for each dimension (`x`, `y`).
  - **Wasserstein Distance:** Measures the overall difference between distributions.

    AS of the provided model: 
        MMD: 0.0019
        KL Divergence: x=0.0858, y=0.0534
        Wasserstein Distances: x: 0.008623, y: 0.003177, r: 0.006073, g: 0.004483, b: 0.011655


- **Visual Analysis:**
  - Histograms: Show alignment of real and generated data.
  - Scatter Plots: Highlight patterns and relationships between dimensions.
  - PCA: Confirms similarity in reduced dimensions.

    Check `plot/` directory for the results.