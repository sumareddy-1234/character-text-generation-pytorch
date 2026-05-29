# Character Text Generation using PyTorch

## Project Overview

This project implements a Character-Level Text Generation system using Deep Learning models in PyTorch.

The main goal of the project is to train neural networks to learn patterns from text data and generate new text one character at a time.

Two different deep learning architectures were implemented and compared:

- LSTM (Long Short-Term Memory)
- Transformer Encoder

The models are trained on text sequences and predict the next character based on previously seen characters.

For example:

Input:
```text
To be or not to b
```

Predicted next character:
```text
e
```

After training, the models can generate new text starting from a seed sentence.

---

# Objectives

- Understand character-level language modeling
- Learn sequence prediction using deep learning
- Implement LSTM architecture in PyTorch
- Implement Transformer architecture in PyTorch
- Train and compare both models
- Generate text using temperature-based sampling
- Visualize training loss curves

---
### 🧪 Dataset Reference

- Tiny Shakespeare Dataset  
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt  
  Used as training corpus for both LSTM and Transformer models.

# Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib

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
- Loads the dataset
- Extracts unique characters
- Creates vocabulary mappings
- Encodes characters into integers

Example:

```text
a → 0
b → 1
c → 2
```

The vocabulary is saved as:

```text
input/vocab.json
```

---

## 2. LSTM Model Implementation

File used:
```text
src/model_lstm.py
```

The LSTM model contains:

- Embedding Layer
- Multi-layer LSTM
- Fully Connected Output Layer

Purpose:
- Learns sequential dependencies in text
- Predicts the next character in a sequence

---

## 3. Transformer Model Implementation

File used:
```text
src/model_transformer.py
```

The Transformer model contains:

- Embedding Layer
- Positional Encoding
- Transformer Encoder Layers
- Fully Connected Layer

Purpose:
- Learns contextual relationships using self-attention
- Generates text based on learned patterns

---

# Model Training

Training script:
```text
src/train.py
```

## Train LSTM Model

```bash
python src/train.py --model lstm --epochs 5
```

## Train Transformer Model

```bash
python src/train.py --model transformer --epochs 5
```

During training:
- Loss values are displayed batch-wise
- Average epoch loss is calculated
- Trained models are saved in `models/`

Example output:

```text
Epoch 5/5 Completed
Average Loss: 1.4465
```

---

# Text Generation

Generation script:
```text
src/generate.py
```

## Generate Text using LSTM

```bash
python src/generate.py --model lstm --model_path models/lstm.pth
```

## Generate Text using Transformer

```bash
python src/generate.py --model transformer --model_path models/transformer.pth
```

The models generate text using:
- Seed text
- Temperature sampling

---

# Temperature Sampling

Temperature controls randomness during text generation.

## Low Temperature (0.5)
- More predictable output
- Safer text generation

Example:
```text
To be or not to be to the lave be platest...
```

## Medium Temperature (1.0)
- Balanced randomness

Example:
```text
To be or not to beore Soll, and not'lf...
```

## High Temperature (1.5)
- More random and creative output

Example:
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
- Produced more readable and structured text
- Learned character patterns gradually
- Performance improved after increasing epochs

## Transformer Model
- Learned faster during training
- Generated repetitive characters in some cases
- Requires better tuning and larger training data for improved results

---

# Challenges Faced and Solutions

## 1. Python and pip Setup Issues

Problem:
```text
'pip' is not recognized as an internal or external command
```

Solution:
Used:

```bash
py -m pip install <package_name>
```

---

## 2. PyTorch DLL Error

Problem:
```text
OSError: WinError 126
```

Cause:
Missing Microsoft Visual C++ Redistributable.

Solution:
Installed the required Visual C++ Redistributable package.

---

## 3. Training Appeared Stuck

Problem:
No output was visible during training.

Solution:
Added batch-level logging inside the training loop to display progress and loss values.

---

## 4. KeyError During Text Generation

Problem:
```text
KeyError while decoding predicted characters
```

Solution:
Corrected dictionary key handling in `generate.py`.

---

## 5. Transformer Generated Repetitive Characters

Problem:
Output contained repeated characters like:

```text
eeeeeeeeeeeeeeee
```

Solution:
- Increased training epochs
- Tested different temperature values
- Compared outputs with the LSTM model

---

## 6. GitHub Ignoring Output Files

Problem:
`loss_curves.png` and model files were not uploaded.

Cause:
`.gitignore` rules ignored `.png` and `.pth` files.

Solution:
Updated `.gitignore` and force-added files using:

```bash
git add -f results/loss_curves.png
```

---

# Learning Outcomes

Through this project, the following concepts were learned:

- Character-level text generation
- Sequence prediction
- Deep learning model training
- LSTM architecture
- Transformer architecture
- PyTorch implementation
- Text sampling techniques
- Model saving and loading
- Git and GitHub workflow
- Debugging training and environment issues

---
# Docker Setup and Execution

This project supports Docker-based execution to ensure a consistent and reproducible environment across different systems.

Docker was used to:

* Simplify dependency management
* Avoid local environment conflicts
* Run the project in an isolated containerized environment
* Ensure consistent execution across machines

---

## Docker Requirements

Before running the project, install the following:

* Docker Desktop
* Docker Compose

Verify installation:

```bash
docker --version
docker compose version
```

---

# Running the Project with Docker

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

* Downloads the required Python image
* Installs project dependencies
* Configures the application container

---

## 3. Verify Docker Environment

```bash
docker compose run --rm app python --version
```

Example output:

```text
Python 3.10.20
```

---

# Training Models using Docker

## Train LSTM Model

```bash
docker compose run --rm app python src/train.py --model lstm --epochs 1
```

---

## Train Transformer Model

```bash
docker compose run --rm app python src/train.py --model transformer --epochs 1
```

During training:

* Batch-wise loss values are displayed
* Average epoch loss is calculated
* Trained models are automatically saved inside the `models/` directory

---

# Generate Text using Docker

## Generate Text using LSTM

```bash
docker compose run --rm app python src/generate.py --model lstm --model_path models/lstm.pth --temperature 1.0
```

---

## Generate Text using Transformer

```bash
docker compose run --rm app python src/generate.py --model transformer --model_path models/transformer.pth --temperature 1.0
```

---

# Docker Project Files

Additional Docker configuration files used in this project:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

## Purpose of Each File

### Dockerfile

Defines:

* Base Python environment
* Required dependencies
* Project execution setup

### docker-compose.yml

Manages:

* Container configuration
* Service execution
* Volume and command handling

### .dockerignore

Excludes unnecessary files from Docker image creation to improve build speed and reduce image size.

---

# Benefits of Docker in This Project

* Portable execution environment
* Simplified setup process
* Consistent dependency management
* Improved reproducibility
* Easier deployment and testing

---

# Docker Build Status

The project was successfully:

* Built using Docker Compose
* Trained inside Docker containers
* Executed using isolated container environments

Both LSTM and Transformer models were successfully trained and tested using Docker.

---

# Conclusion

This project successfully demonstrates character-level text generation using LSTM and Transformer models in PyTorch.

The project helped in understanding:
- How neural networks learn text patterns
- Differences between LSTM and Transformer architectures
- Training workflows for sequence models
- Practical debugging and implementation challenges in deep learning projects

---

# Author

Suma Satti
