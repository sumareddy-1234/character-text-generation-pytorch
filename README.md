# Character Text Generation using PyTorch

## Project Overview

This project implements a Character-Level Text Generation system using Deep Learning models in PyTorch.

The primary objective of the project is to train neural networks to learn patterns from textual data and generate new text one character at a time.

Two different deep learning architectures were implemented and compared:

* LSTM (Long Short-Term Memory)
* Transformer Encoder

The models are trained on character sequences and predict the next character based on previously observed text patterns.

### Example

Input:

```text
To be or not to b
```

Predicted next character:

```text
e
```

After training, the models can generate entirely new text from a given seed sentence.

---

# Objectives

* Understand character-level language modeling
* Learn sequence prediction using deep learning
* Implement LSTM architecture using PyTorch
* Implement Transformer architecture using PyTorch
* Train and compare both models
* Generate text using temperature-based sampling
* Visualize training loss curves
* Containerize the project using Docker

---

# Dataset Reference

* Tiny Shakespeare Dataset
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The dataset was used as the training corpus for both LSTM and Transformer models.

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Docker
* Docker Compose
* Git & GitHub

---

# Project Structure

```text
character-text-generation/
│
├── input/                     # Dataset and vocabulary files
│
├── models/                    # Saved trained model files
│   ├── lstm.pth
│   └── transformer.pth
│
├── results/                   # Output results
│   ├── generated_samples.json
│   ├── loss_curves.png
│   ├── lstm_loss.json
│   └── transformer_loss.json
│
├── src/                       # Source code
│   ├── prepare_data.py
│   ├── model_lstm.py
│   ├── model_transformer.py
│   ├── train.py
│   ├── generate.py
│   └── plot_results.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

---

# Project Workflow

## 1. Data Preparation

File used:

```text
src/prepare_data.py
```

This script:

* Loads the dataset
* Extracts unique characters
* Creates vocabulary mappings
* Encodes characters into integers

### Example

```text
a → 0
b → 1
c → 2
```

The vocabulary is stored in:

```text
input/vocab.json
```

---

## 2. LSTM Model Implementation

File used:

```text
src/model_lstm.py
```

The LSTM model consists of:

* Embedding Layer
* Multi-layer LSTM
* Fully Connected Output Layer

### Purpose

* Learns sequential dependencies in text
* Predicts the next character in a sequence

---

## 3. Transformer Model Implementation

File used:

```text
src/model_transformer.py
```

The Transformer model consists of:

* Embedding Layer
* Positional Encoding
* Transformer Encoder Layers
* Fully Connected Layer

### Purpose

* Learns contextual relationships using self-attention
* Generates text based on learned character patterns

---

# Docker Setup and Execution

This project supports Docker-based execution to provide a consistent and reproducible environment across different systems.

Docker was used to:

* Simplify dependency management
* Avoid local environment conflicts
* Ensure consistent execution across machines
* Run the project in an isolated containerized environment

---

## Docker Requirements

Install the following before running the project:

* Docker Desktop
* Docker Compose

Verify installation:

```bash
docker --version
docker compose version
```

---

# Running the Project using Docker

## 1. Clone the Repository

```bash
git clone https://github.com/sumareddy-1234/character-text-generation-pytorch.git
cd character-text-generation-pytorch
```

---

## 2. Build Docker Image

```bash
docker compose build
```

This command:

* Downloads the required Python base image
* Installs project dependencies
* Configures the application container

---

## 3. Verify Docker Environment

```bash
docker compose run --rm app python --version
```

### Example Output

```text
Python 3.10.20
```

---

# Model Training

Training script:

```text
src/train.py
```

## Train LSTM Model

### Local Execution

```bash
python src/train.py --model lstm --epochs 5
```

### Docker Execution

```bash
docker compose run --rm app python src/train.py --model lstm --epochs 1
```

---

## Train Transformer Model

### Local Execution

```bash
python src/train.py --model transformer --epochs 5
```

### Docker Execution

```bash
docker compose run --rm app python src/train.py --model transformer --epochs 1
```

---

During training:

* Batch-wise loss values are displayed
* Average epoch loss is calculated
* Trained models are saved inside the `models/` directory

### Example Output

```text
Epoch 1/1 Completed
Average Loss: 2.1329
```

---

# Text Generation

Generation script:

```text
src/generate.py
```

## Generate Text using LSTM

### Local Execution

```bash
python src/generate.py --model lstm --model_path models/lstm.pth
```

### Docker Execution

```bash
docker compose run --rm app python src/generate.py --model lstm --model_path models/lstm.pth --temperature 1.0
```

---

## Generate Text using Transformer

### Local Execution

```bash
python src/generate.py --model transformer --model_path models/transformer.pth
```

### Docker Execution

```bash
docker compose run --rm app python src/generate.py --model transformer --model_path models/transformer.pth --temperature 1.0
```

---

# Temperature Sampling

Temperature controls randomness during text generation.

## Low Temperature (0.5)

* Produces more predictable output
* Safer and structured text generation

### Example

```text
To be or not to be to the lave be platest...
```

---

## Medium Temperature (1.0)

* Produces balanced randomness

### Example

```text
To be or not to beore Soll, and not'lf...
```

---

## High Temperature (1.5)

* Produces highly random and creative output

### Example

```text
To be or not to bephasgedFela abkade...
```

---

# Results

Generated outputs are stored in:

```text
results/generated_samples.json
```

Training losses are stored in:

```text
results/lstm_loss.json
results/transformer_loss.json
```

Loss curve visualization:

```text
results/loss_curves.png
```

---

# Observations

## LSTM Model

* Produced more readable and structured text
* Learned character patterns gradually
* Performance improved after increasing epochs

## Transformer Model

* Learned faster during training
* Generated repetitive characters in some cases
* Requires better tuning and larger training data for improved results

---

# Challenges Faced and Solutions

## 1. Python and pip Setup Issues

### Problem

```text
'pip' is not recognized as an internal or external command
```

### Solution

Used:

```bash
py -m pip install <package_name>
```

---

## 2. PyTorch DLL Error

### Problem

```text
OSError: WinError 126
```

### Cause

Missing Microsoft Visual C++ Redistributable.

### Solution

Installed the required Visual C++ Redistributable package.

---

## 3. Training Appeared Stuck

### Problem

No visible output during training.

### Solution

Added batch-level logging inside the training loop to display progress and loss values.

---

## 4. KeyError During Text Generation

### Problem

```text
KeyError while decoding predicted characters
```

### Solution

Corrected dictionary key handling in `generate.py`.

---

## 5. Transformer Generated Repetitive Characters

### Problem

Output contained repeated characters such as:

```text
eeeeeeeeeeeeeeee
```

### Solution

* Increased training epochs
* Tested different temperature values
* Compared generated outputs with the LSTM model

---

## 6. GitHub Ignoring Output Files

### Problem

`loss_curves.png` and model files were not uploaded.

### Cause

`.gitignore` rules ignored `.png` and `.pth` files.

### Solution

Updated `.gitignore` and force-added files using:

```bash
git add -f results/loss_curves.png
```

---

## 7. Docker Build and Dependency Setup

### Problem

Initial Docker setup failed because dependencies could not be downloaded properly.

### Solution

* Rebuilt the Docker image using Docker Compose
* Verified internet connectivity and Docker Desktop configuration
* Successfully installed all dependencies inside the container environment

---

# Learning Outcomes

Through this project, the following concepts were learned:

* Character-level text generation
* Sequence prediction
* Deep learning model training
* LSTM architecture
* Transformer architecture
* PyTorch implementation
* Text sampling techniques
* Docker containerization
* Model saving and loading
* Git and GitHub workflow
* Debugging training and environment issues

---

# Conclusion

This project successfully demonstrates character-level text generation using both LSTM and Transformer architectures in PyTorch.

The project provided practical experience in:

* Deep learning model implementation
* Sequence modeling
* Text generation techniques
* Docker-based deployment and execution
* Model training and evaluation workflows
* Debugging real-world development issues

The comparison between LSTM and Transformer models helped in understanding how different neural network architectures learn textual patterns and generate language.

---

# Author

Suma Satti
