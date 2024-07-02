# Seq-to-Seq Transformer Model from Scratch in PyTorch

This project implements a Transformer model from scratch using PyTorch. It includes a training script to train the model on a custom dataset and supports distributed training.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
- [Model Architecture](#model-architecture)
  - [Encoder](#encoder)
  - [Decoder](#decoder)

## Features

- Transformer model implementation from scratch
- Customizable model parameters
- Training script to train the model
- Easy-to-follow code structure

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Training
To train the Transformer model, run the training script with a configuration file. The configuration file allows you to specify all necessary parameters for training.

You can use the `config.json` file where you provide the following parameters:

- `train_data`: Path to the training dataset. It can be on your local machine or on HuggingFace.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for the optimizer.
- `valid_data`: Path to the validation dataset.
- `model_dim`: Dimensionality of the model.
- `hidden_dims`: Dimensionality of the hidden layers.
- `num_heads`: Number of attention heads.
- `num_layers`: Number of encoder/decoder layers.
- `dropout`: Dropout rate.
- `max_seq_length`: Maximum sequence length.

The provided training script is set up for language translation using the `wmt14` dataset and the tokenizer of the `Helsinki-NLP/opus-mt-en-de` model.

### Distributed Training
The training script supports distributed training using `Fully Sharded Data Parallel` (FSDP) with Hugging Face's `accelerate` library. This allows you to efficiently train large models across multiple GPUs or nodes.

The settings for accelerate are found in `accelerate_config.yaml`.
To start the training you can use the CLI command:
```bash
accelerate launch --config-file accelerate_config.yaml train.py
```

## Model Architecture
The Transformer model consists of an encoder and a decoder.

Encoder: Processes the input sequence and produces a representation of it. </br>
Decoder: Takes the encoder's representation and generates the output sequence.

### Encoder

The Encoder in a Transformer model is responsible for processing the input sequence and generating a contextualized representation of it. The Encoder consists of several layers, each containing two main sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. Each sub-layer is followed by layer normalization and a residual connection.

#### Components of the Encoder

1. **Input Embedding**:
   - Converts token indices into dense vectors of fixed size (model dimension).
   - Includes positional encoding to provide information about the position of each token in the sequence.

2. **Multi-Head Self-Attention Mechanism**:
   - **Self-Attention**: Allows the model to focus on different parts of the input sequence, capturing dependencies between words regardless of their distance.
   - **Multi-Head**: Multiple attention mechanisms run in parallel (heads), each with its own set of parameters, enabling the model to capture various relationships within the input.

3. **Position-Wise Feed-Forward Network**:
   - A two-layer fully connected network applied independently to each position, with a non-linear activation function (GELU) in between.
   - Applies a non-linear transformation to the attention outputs.

4. **Layer Normalization and Residual Connections**:
   - **Layer Normalization**: Normalizes the output of each sub-layer to stabilize and accelerate training.
   - **Residual Connections**: Adds the input of the sub-layer to its output, ensuring the retention of original information and aiding gradient flow during backpropagation.

5. **Dropout**:
   - A regularization technique to prevent overfitting by randomly setting a fraction of input units to zero during training.


### Decoder

The Decoder in a Transformer model generates the output sequence based on the encoded representation of the input sequence and the previously generated tokens. The Decoder consists of several layers, each containing three main sub-layers: a masked multi-head self-attention mechanism, a multi-head attention mechanism over the encoder's output, and a position-wise fully connected feed-forward network. Each sub-layer is followed by layer normalization and a residual connection.

#### Components of the Decoder

1. **Input Embedding**:
   - Converts token indices into dense vectors of fixed size (model dimension).
   - Includes positional encoding to provide information about the position of each token in the sequence.

2. **Masked Multi-Head Self-Attention Mechanism**:
   - **Self-Attention**: Allows the model to focus on different parts of the decoder input sequence, capturing dependencies between words regardless of their distance.
   - **Masked**: Ensures that the prediction for a particular position depends only on the known outputs at preceding positions, preventing information leakage.

3. **Multi-Head Attention Mechanism over Encoder Output**:
   - Uses the encoder's output as keys and values and the decoder's output as queries to focus on relevant parts of the input sequence.

4. **Position-Wise Feed-Forward Network**:
   - A two-layer fully connected network applied independently to each position, with a non-linear activation function (GELU) in between.
   - Applies a non-linear transformation to the attention outputs.

5. **Layer Normalization and Residual Connections**:
   - **Layer Normalization**: Normalizes the output of each sub-layer to stabilize and accelerate training.
   - **Residual Connections**: Adds the input of the sub-layer to its output, ensuring the retention of original information and aiding gradient flow during backpropagation.