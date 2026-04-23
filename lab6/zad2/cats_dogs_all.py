import argparse
import json
import random
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
CLASS_TO_INDEX = {"cat": 0, "dog": 1}
INDEX_TO_CLASS = {0: "cat", 1: "dog"}


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def unzip_dataset(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def build_dataframe(images_dir: Path):
    rows = []

    for image_path in sorted(images_dir.glob("*.jpg")):
        name = image_path.name.lower()

        if name.startswith("cat."):
            label = "cat"
        elif name.startswith("dog."):
            label = "dog"
        else:
            continue

        rows.append(
            {
                "file_path": str(image_path),
                "label": label,
            }
        )

    df = pd.DataFrame(rows)
    return df


def split_dataframe(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def get_transforms(image_size=64, augment=False, pretrained=False):
    from torchvision import transforms

    if pretrained:
        train_list = [
            transforms.Resize((224, 224)),
        ]
        if augment:
            train_list.append(transforms.RandomHorizontalFlip())
            train_list.append(transforms.RandomRotation(10))
        train_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        train_transform = transforms.Compose(train_list)
        return train_transform, val_transform

    train_list = [
        transforms.Resize((image_size, image_size)),
    ]

    if augment:
        train_list.append(transforms.RandomHorizontalFlip())
        train_list.append(transforms.RandomRotation(10))

    train_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_transform = transforms.Compose(train_list)

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    return train_transform, val_transform


class CatsDogsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        label = CLASS_TO_INDEX[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, row["file_path"]


def make_loaders(train_df, val_df, test_df, batch_size=64, image_size=64, augment=False, pretrained=False):
    train_transform, val_transform = get_transforms(
        image_size=image_size,
        augment=augment,
        pretrained=pretrained,
    )

    train_dataset = CatsDogsDataset(train_df, transform=train_transform)
    val_dataset = CatsDogsDataset(val_df, transform=val_transform)
    test_dataset = CatsDogsDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class SimpleCNN(nn.Module):
    def __init__(self, filters=(32, 64, 128), activation_name="relu", dropout=0.0):
        super().__init__()

        activation = get_activation(activation_name)

        self.features = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
            activation,
            nn.MaxPool2d(2),

            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            activation,
            nn.MaxPool2d(2),

            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            activation,
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[2] * 8 * 8, 128),
            activation,
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_transfer_model():
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)

    return avg_loss, acc


def evaluate_loader(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []
    paths = []

    with torch.no_grad():
        for images, labels, batch_paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
            paths.extend(list(batch_paths))

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)

    return avg_loss, acc, y_true, y_pred, y_prob, paths


def save_history_plot(history_df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["cat", "dog"],
        ax=ax,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_misclassified_examples(predictions_df: pd.DataFrame, output_path: Path, how_many=16):
    wrong_df = predictions_df[predictions_df["true_label"] != predictions_df["pred_label"]].copy()

    if len(wrong_df) == 0:
        return

    wrong_df = wrong_df.sort_values("confidence", ascending=False).head(how_many)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = (len(wrong_df) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), dpi=160)
    if hasattr(axes, "flatten"):
        axes_list = list(axes.flatten())
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.axis("off")

    for ax, (_, row) in zip(axes_list, wrong_df.iterrows(), strict=False):
        image = Image.open(row["file_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"true={row['true_label']}\npred={row['pred_label']}\nconf={row['confidence']:.3f}",
            fontsize=8,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_and_save(model, train_loader, val_loader, experiment_dir: Path, epochs, lr, optimizer_name, device):
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    history_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _, _ = evaluate_loader(model, val_loader, criterion, device)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(experiment_dir / "history.csv", index=False)

    if best_state is not None:
        torch.save(best_state, experiment_dir / "best_model.pt")
        model.load_state_dict(best_state)

    save_history_plot(history_df, experiment_dir / "learning_curves.png")

    return model, history_df, best_val_acc


def build_predictions_df(y_true, y_pred, y_prob, paths):
    rows = []

    for true_idx, pred_idx, probs, path in zip(y_true, y_pred, y_prob, paths, strict=True):
        rows.append(
            {
                "file_path": path,
                "true_label": INDEX_TO_CLASS[true_idx],
                "pred_label": INDEX_TO_CLASS[pred_idx],
                "confidence": max(probs),
                "prob_cat": probs[0],
                "prob_dog": probs[1],
            }
        )

    return pd.DataFrame(rows)


def run_custom_experiment(name, cfg, train_df, val_df, test_df, device, outputs_root: Path):
    print("=" * 80)
    print("start eksperymentu:", name)

    experiment_dir = outputs_root / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        train_df,
        val_df,
        test_df,
        batch_size=cfg["batch_size"],
        image_size=cfg["image_size"],
        augment=cfg["augment"],
        pretrained=False,
    )

    model = SimpleCNN(
        filters=cfg["filters"],
        activation_name=cfg["activation"],
        dropout=cfg["dropout"],
    ).to(device)

    model, history_df, best_val_acc = train_and_save(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_dir=experiment_dir,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        optimizer_name=cfg["optimizer"],
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred, y_prob, paths = evaluate_loader(
        model, test_loader, criterion, device
    )

    y_true_labels = [INDEX_TO_CLASS[x] for x in y_true]
    y_pred_labels = [INDEX_TO_CLASS[x] for x in y_pred]

    predictions_df = build_predictions_df(y_true, y_pred, y_prob, paths)
    predictions_df.to_csv(experiment_dir / "test_predictions.csv", index=False)

    save_confusion_matrix_plot(y_true_labels, y_pred_labels, experiment_dir / "confusion_matrix.png")
    save_misclassified_examples(predictions_df, experiment_dir / "misclassified_examples.png")

    wrong_total = int((predictions_df["true_label"] != predictions_df["pred_label"]).sum())
    wrong_cats_as_dogs = int(
        ((predictions_df["true_label"] == "cat") & (predictions_df["pred_label"] == "dog")).sum()
    )
    wrong_dogs_as_cats = int(
        ((predictions_df["true_label"] == "dog") & (predictions_df["pred_label"] == "cat")).sum()
    )

    metrics = {
        "experiment_name": name,
        "model_type": "custom_cnn",
        "optimizer": cfg["optimizer"],
        "activation": cfg["activation"],
        "filters": list(cfg["filters"]),
        "dropout": cfg["dropout"],
        "augment": cfg["augment"],
        "epochs": cfg["epochs"],
        "learning_rate": cfg["lr"],
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "wrong_total": wrong_total,
        "wrong_cats_as_dogs": wrong_cats_as_dogs,
        "wrong_dogs_as_cats": wrong_dogs_as_cats,
    }

    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("koniec eksperymentu:", name)
    print(metrics)

    return metrics


def run_transfer_experiment(name, epochs, batch_size, lr, train_df, val_df, test_df, device, outputs_root: Path):
    print("=" * 80)
    print("start eksperymentu transfer learning:", name)

    experiment_dir = outputs_root / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        train_df,
        val_df,
        test_df,
        batch_size=batch_size,
        image_size=224,
        augment=True,
        pretrained=True,
    )

    model = build_transfer_model().to(device)

    model, history_df, best_val_acc = train_and_save(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_dir=experiment_dir,
        epochs=epochs,
        lr=lr,
        optimizer_name="adam",
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred, y_prob, paths = evaluate_loader(
        model, test_loader, criterion, device
    )

    y_true_labels = [INDEX_TO_CLASS[x] for x in y_true]
    y_pred_labels = [INDEX_TO_CLASS[x] for x in y_pred]

    predictions_df = build_predictions_df(y_true, y_pred, y_prob, paths)
    predictions_df.to_csv(experiment_dir / "test_predictions.csv", index=False)

    save_confusion_matrix_plot(y_true_labels, y_pred_labels, experiment_dir / "confusion_matrix.png")
    save_misclassified_examples(predictions_df, experiment_dir / "misclassified_examples.png")

    wrong_total = int((predictions_df["true_label"] != predictions_df["pred_label"]).sum())
    wrong_cats_as_dogs = int(
        ((predictions_df["true_label"] == "cat") & (predictions_df["pred_label"] == "dog")).sum()
    )
    wrong_dogs_as_cats = int(
        ((predictions_df["true_label"] == "dog") & (predictions_df["pred_label"] == "cat")).sum()
    )

    metrics = {
        "experiment_name": name,
        "model_type": "transfer_resnet18",
        "optimizer": "adam",
        "activation": "resnet18_pretrained",
        "filters": [],
        "dropout": 0.0,
        "augment": True,
        "epochs": epochs,
        "learning_rate": lr,
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "wrong_total": wrong_total,
        "wrong_cats_as_dogs": wrong_cats_as_dogs,
        "wrong_dogs_as_cats": wrong_dogs_as_cats,
    }

    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("koniec eksperymentu transfer:", name)
    print(metrics)

    return metrics


def save_dataset_info(df, train_df, val_df, test_df, outputs_root: Path):
    outputs_root.mkdir(parents=True, exist_ok=True)

    info = {
        "all_count": int(len(df)),
        "cat_count": int((df["label"] == "cat").sum()),
        "dog_count": int((df["label"] == "dog").sum()),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
    }

    with open(outputs_root / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=str, default="dogs-cats-mini.zip")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "custom", "transfer"])
    parser.add_argument("--epochs-custom", type=int, default=5)
    parser.add_argument("--epochs-transfer", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    set_seed(SEED)

    zip_path = Path(args.zip_path)
    extracted_root = Path("data_unzipped")
    outputs_root = Path("outputs")
    images_dir = extracted_root / "dogs-cats-mini"

    unzip_dataset(zip_path, extracted_root)

    df = build_dataframe(images_dir)
    train_df, val_df, test_df = split_dataframe(df)

    save_dataset_info(df, train_df, val_df, test_df, outputs_root)

    device = get_device()
    print("device:", device)

    results = []

    custom_configs = {
        "cnn_adam_relu_basic": {
            "optimizer": "adam",
            "activation": "relu",
            "filters": (32, 64, 128),
            "dropout": 0.0,
            "augment": False,
            "epochs": args.epochs_custom,
            "lr": 0.001,
            "batch_size": args.batch_size,
            "image_size": 64,
        },
        "cnn_sgd_relu_basic": {
            "optimizer": "sgd",
            "activation": "relu",
            "filters": (32, 64, 128),
            "dropout": 0.0,
            "augment": False,
            "epochs": args.epochs_custom,
            "lr": 0.01,
            "batch_size": args.batch_size,
            "image_size": 64,
        },
        "cnn_adam_relu_dropout": {
            "optimizer": "adam",
            "activation": "relu",
            "filters": (32, 64, 128),
            "dropout": 0.3,
            "augment": False,
            "epochs": args.epochs_custom,
            "lr": 0.001,
            "batch_size": args.batch_size,
            "image_size": 64,
        },
        "cnn_adam_leaky_aug_big": {
            "optimizer": "adam",
            "activation": "leaky_relu",
            "filters": (32, 64, 256),
            "dropout": 0.3,
            "augment": True,
            "epochs": args.epochs_custom,
            "lr": 0.001,
            "batch_size": args.batch_size,
            "image_size": 64,
        },
    }

    if args.mode in ["all", "custom"]:
        for name, cfg in custom_configs.items():
            metrics = run_custom_experiment(
                name=name,
                cfg=cfg,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                device=device,
                outputs_root=outputs_root,
            )
            results.append(metrics)

    if args.mode in ["all", "transfer"]:
        metrics = run_transfer_experiment(
            name="transfer_resnet18",
            epochs=args.epochs_transfer,
            batch_size=args.batch_size,
            lr=0.0003,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            device=device,
            outputs_root=outputs_root,
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("test_acc", ascending=False)
        results_df.to_csv(outputs_root / "summary.csv", index=False)
        print(results_df)


if __name__ == "__main__":
    main()
