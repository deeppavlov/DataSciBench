import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model_definition import SimpleNeuralNetwork

SEED = 42
HIDDEN_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 100
FINE_TUNE_EPOCHS = 30
PRUNE_RATIO = 0.25

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_split(path):
    frame = pd.read_csv(path)
    features = torch.tensor(frame.drop(columns=["target"]).values.astype(np.float32))
    targets = torch.tensor(frame["target"].values.astype(np.float32)).view(-1, 1)
    return features, targets


def train(model, features, targets, epochs, masks=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(features.size(0))
        total_loss = 0.0
        for start in range(0, features.size(0), BATCH_SIZE):
            index = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(features[index]), targets[index])
            loss.backward()
            optimizer.step()
            if masks is not None:
                apply_masks(model, masks)
            total_loss += loss.item() * index.numel()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss {total_loss / features.size(0):.4f}")
    return model


def evaluate(model, features, targets):
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features))
    mse = float(((probabilities - targets) ** 2).mean())
    accuracy = float(((probabilities > 0.5).float() == targets).float().mean())
    return mse, accuracy


def l1_norms(model):
    return pd.DataFrame(
        [
            {"Layer": name, "L1 Norm": float(param.detach().abs().sum())}
            for name, param in model.named_parameters()
            if "weight" in name
        ]
    )


def prune(model, ratio):
    masks = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            magnitudes = param.detach().abs()
            count = int(ratio * magnitudes.numel())
            threshold = torch.sort(magnitudes.flatten()).values[count - 1]
            mask = (magnitudes > threshold).float()
            param.mul_(mask)
            masks[name] = mask
    return masks


def apply_masks(model, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])


def main():
    x_train, y_train = load_split("../train.csv")
    x_test, y_test = load_split("../test.csv")

    baseline = float(((y_train.mean() - y_test) ** 2).mean())
    print(f"baseline mean-prediction test mse {baseline:.4f}")

    model = SimpleNeuralNetwork(x_train.size(1), HIDDEN_SIZE, 1)
    train(model, x_train, y_train, TRAIN_EPOCHS)
    torch.save(model.state_dict(), "trained_model.pth")
    print("trained mse/acc train {:.4f}/{:.4f} test {:.4f}/{:.4f}".format(*evaluate(model, x_train, y_train), *evaluate(model, x_test, y_test)))

    norms = l1_norms(model)
    norms.to_csv("l1_norms.csv", index=False)
    print(norms)

    masks = prune(model, PRUNE_RATIO)
    torch.save(model.state_dict(), "pruned_model.pth")
    kept = sum(float(mask.sum()) for mask in masks.values())
    total = sum(mask.numel() for mask in masks.values())
    print(f"pruned {1 - kept / total:.4f} of {total} weights")
    print("pruned mse/acc train {:.4f}/{:.4f} test {:.4f}/{:.4f}".format(*evaluate(model, x_train, y_train), *evaluate(model, x_test, y_test)))

    train(model, x_train, y_train, FINE_TUNE_EPOCHS, masks)
    torch.save(model.state_dict(), "fine_tuned_model.pth")
    print("fine-tuned mse/acc train {:.4f}/{:.4f} test {:.4f}/{:.4f}".format(*evaluate(model, x_train, y_train), *evaluate(model, x_test, y_test)))


if __name__ == "__main__":
    main()
