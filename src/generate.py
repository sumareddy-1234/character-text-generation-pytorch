import os
import json
import argparse

import torch
import torch.nn.functional as F

from prepare_data import load_data
from model_lstm import LSTMModel
from model_transformer import TransformerModel


def generate_text(
    model,
    seed_text,
    char_to_int,
    int_to_char,
    length=200,
    temperature=1.0,
    model_type="lstm"
):

    model.eval()

    generated = seed_text

    chars = list(seed_text)

    for _ in range(length):

        encoded = [
            char_to_int.get(c, 0)
            for c in chars[-50:]
        ]

        input_tensor = torch.tensor(
            [encoded],
            dtype=torch.long
        )

        with torch.no_grad():

            if model_type == "lstm":

                hidden = model.init_hidden(1)

                output, hidden = model(
                    input_tensor,
                    hidden
                )

            else:

                output = model(input_tensor)

        logits = output[-1] / temperature

        probabilities = F.softmax(
            logits,
            dim=0
        )

        predicted_index = torch.multinomial(
            probabilities,
            1
        ).item()

        predicted_char = int_to_char[
           predicted_index
        ]
        generated += predicted_char

        chars.append(predicted_char)

    return generated


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "transformer"]
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--seed_text",
        type=str,
        default="To be or not to be"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    text, encoded, char_to_int, int_to_char = load_data(
        "input/shakespeare.txt"
    )

    vocab_size = len(char_to_int)

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

    model.load_state_dict(
        torch.load(args.model_path)
    )

    generated_text = generate_text(
        model=model,
        seed_text=args.seed_text,
        char_to_int=char_to_int,
        int_to_char=int_to_char,
        temperature=args.temperature,
        model_type=args.model
    )

    print("\nGenerated Text:\n")

    print(generated_text)


if __name__ == "__main__":

    main()