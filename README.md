# llm-from-scratch

A high-performance, transformer-based language model implemented entirely from fundamental building blocks. This project focuses on the granular implementation of modern LLM architectures, moving away from high-level abstractions to master the underlying mechanics of large language models.

## 🚀 Overview

This repository contains a complete implementation of a Transformer-based LLM. The core philosophy of this project is **zero reliance on high-level `torch.nn` layers** (like `nn.Linear`, `nn.LayerNorm`, or `nn.Transformer`). 

Instead, every component is built using `torch.nn.Parameter` and raw tensor operations to ensure a deep understanding of the forward and backward passes.

### Key Technical Features

* **Model Architecture:**
    * **Pre-normalization:** Improved training stability by normalizing before blocks.
    * **RMSNorm:** Root Mean Square Layer Normalization for faster computation.
    * **SwiGLU Activation:** Implementation of the Gated Linear Unit variant used in Llama architectures.
    * **Rotary Positional Embeddings (RoPE):** Features an auto-expanding implementation for flexible sequence lengths.
* **Training & Optimization:**
    * **Custom AdamW:** Built from the `torch.optim.Optimizer` base class.
    * **Learning Rate Schedule:** Cosine annealing for optimal convergence.
    * **Gradient Clipping:** To prevent exploding gradients during intense training phases.

## 🏗 Architecture

The model strictly follows the architectural flow illustrated below:

![Architecture](img/architecture.png)


## 📂 Project Structure

```text
├── main/
│   ├── model.py                # Core Transformer component implementations
│   ├── tokenizer_optimized.py   # Custom BPE/Tokenization
│   ├── train_model.py          # Training utils and optimizer
│   └── run_train_model.py      # Entry point for training
├── tokenized_data/             # Pre-processed datasets for training
├── trained_tokenizer/          # Saved tokenizer states
├── img/                        # Architectural diagrams
├── run.sh                      # Shell script for one-touch execution
└── generate_tree.py            # Utility for project visualization
```

## 🛠 Usage (to train it urself)

This implementation is optimized for efficiency and is fully trainable on consumer hardware, including MacBook Pro (M-series) chips.

### Quick Start

To begin training on whatever dataset you like:

1. Clone the repository.
2. Ensure you have PyTorch installed.
3. Implement the tokenizer to fill in `trained_tokenizer` and `tokenized_data`.
4. Execute the training script (you may need to adjust some params a little before running):
    ```sh
    ./run.sh
    ```

## 🧪 Implementation Constraints

To demonstrate technical rigor, this project intentionally avoids `torch.nn` high-level definitions. The only components used from `torch` are:

- `torch.nn.Parameter`: For weight initialization.
- Container classes: (`Module`, `ModuleList`, `Sequential`) for organizing the model graph.
- `torch.optim.Optimizer`: Used only as a base class for a ground-up AdamW implementation.

## 🙏 Ackowledgements

- Stanford University CS336: A profound thank you to the course instructors and material for the guidance and motivation required to implement these complex systems from the ground up.
- Xuying Li: For the excellent recommendation of the CS336 curriculum.

License: MIT
