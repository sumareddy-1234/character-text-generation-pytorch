import json
import torch


def load_data(path):

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))

    char_to_int = {
        ch: i for i, ch in enumerate(chars)
    }

    int_to_char = {
        i: ch for ch, i in char_to_int.items()
    }

    encoded = [
        char_to_int[ch]
        for ch in text
    ]

    return text, encoded, char_to_int, int_to_char


def create_sequences(encoded, seq_length=100):

    x = []
    y = []

    for i in range(len(encoded) - seq_length):

        x.append(encoded[i:i + seq_length])

        y.append(encoded[i + 1:i + seq_length + 1])

    return torch.tensor(x), torch.tensor(y)


if __name__ == "__main__":

    text, encoded, c2i, i2c = load_data(
        "input/shakespeare.txt"
    )

    with open("input/vocab.json", "w") as f:

        json.dump(
            {
                "char_to_int": c2i,
                "int_to_char": i2c
            },
            f
        )

    print("Dataset loaded successfully")
    print("Vocabulary Size:", len(c2i))