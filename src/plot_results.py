import json
import matplotlib.pyplot as plt


with open("results/lstm_loss.json", "r") as f:
    lstm_loss = json.load(f)

with open("results/transformer_loss.json", "r") as f:
    transformer_loss = json.load(f)


plt.figure(figsize=(8, 5))


plt.plot(
    range(1, len(lstm_loss) + 1),
    lstm_loss,
    marker='o',
    label="LSTM"
)

plt.plot(
    range(1, len(transformer_loss) + 1),
    transformer_loss,
    marker='o',
    label="Transformer"
)

plt.title("Training Loss Comparison")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.xticks(range(1, max(len(lstm_loss), len(transformer_loss)) + 1))

plt.legend()

plt.grid(True)

plt.savefig("results/loss_curves.png")

print("Loss curve saved successfully")