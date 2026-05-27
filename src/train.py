import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import load_data, create_sequences
from model_lstm import LSTMModel
from model_transformer import TransformerModel


def train():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "transformer"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    args = parser.parse_args()

    text, encoded, char_to_int, int_to_char = load_data(
        "input/shakespeare.txt"
    )

    # Reduce dataset size for faster CPU training
    encoded = encoded[:50000]

    x, y = create_sequences(
        encoded,
        seq_length=50
    )

    vocab_size = len(char_to_int)

    batch_size = 16

    if args.model == "lstm":

        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            n_layers=2
        )

    else:

        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=64,
            nhead=2,
            num_layers=2
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    losses = []

    print(f"\nTraining {args.model.upper()} model...\n")

    for epoch in range(args.epochs):

        total_loss = 0

        for i in range(0, len(x), batch_size):

            inputs = x[i:i + batch_size]

            targets = y[i:i + batch_size]

            optimizer.zero_grad()

            if args.model == "lstm":

                hidden = model.init_hidden(
                    inputs.size(0)
                )

                outputs, hidden = model(
                    inputs,
                    hidden
                )

            else:

                outputs = model(inputs)

            loss = criterion(
                outputs,
                targets.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1
            )

            optimizer.step()

            total_loss += loss.item()

            if i % 1600 == 0:

                print(
                    f"Batch {i}/{len(x)} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / (len(x) // batch_size)

        losses.append(avg_loss)

        print(
            f"\nEpoch {epoch+1}/{args.epochs} Completed"
        )

        print(
            f"Average Loss: {avg_loss:.4f}\n"
        )

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        f"models/{args.model}.pth"
    )

    os.makedirs("results", exist_ok=True)

    with open(
        f"results/{args.model}_loss.json",
        "w"
    ) as f:

        json.dump(losses, f)

    print("Training completed successfully")

    print(
        f"Model saved at models/{args.model}.pth"
    )


if __name__ == "__main__":

    train()