import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import AdamW

from src.config_v2 import (
    DATASET_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VAL_SPLIT,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    LAST_MODEL_NAME,
)

from src.dataset import RVLCDIPDataset
from src.dataloader import get_dataloaders
from src.model_v2 import build_model
from src.trainer import train_one_epoch, validate_one_epoch


def main():

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Using Device : {device}")
    print("=" * 60)

    # -------------------------------------------------
    # Transform
    # -------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = RVLCDIPDataset(
        root_dir=DATASET_PATH,
        transform=transform
    )

    print(f"Total Images : {len(dataset)}")
    print(f"Classes : {dataset.classes}")

    # -------------------------------------------------
    # DataLoader
    # -------------------------------------------------

    train_loader, val_loader = get_dataloaders(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        num_workers=NUM_WORKERS,
        seed=SEED
    )

    print(f"Training Batches : {len(train_loader)}")
    print(f"Validation Batches : {len(val_loader)}")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = build_model().to(device)

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # -------------------------------------------------
    # Scheduler
    # -------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2
    )

    # -------------------------------------------------
    # Checkpoints
    # -------------------------------------------------

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        BEST_MODEL_NAME
    )

    last_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        LAST_MODEL_NAME
    )

    start_epoch = 0
    best_acc = 0.0

    # -------------------------------------------------
    # Resume Training
    # -------------------------------------------------

    checkpoint_path = None

    if os.path.exists(last_checkpoint):
        checkpoint_path = last_checkpoint
        print("=" * 60)
        print("Resuming from LAST checkpoint")
        print("=" * 60)

    elif os.path.exists(best_checkpoint):
        checkpoint_path = best_checkpoint
        print("=" * 60)
        print("LAST checkpoint not found.")
        print("Resuming from BEST checkpoint")
        print("=" * 60)

    if checkpoint_path is not None:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        best_acc = checkpoint["best_accuracy"]

        print(f"Starting from Epoch {start_epoch + 1}")
        print(f"Best Accuracy : {best_acc:.2f}%")

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    for epoch in range(start_epoch, EPOCHS):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("=" * 60)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nTrain Loss : {train_loss:.4f}")
        print(f"Train Accuracy : {train_acc:.2f}%")
        print(f"Validation Loss : {val_loss:.4f}")
        print(f"Validation Accuracy : {val_acc:.2f}%")
        print(f"Learning Rate : {current_lr:.6f}")

        # ---------------------------------------------
        # Save Best Model
        # ---------------------------------------------

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_accuracy": best_acc,
                },
                best_checkpoint
            )

            print("\n✅ Best Model Saved!")

        # ---------------------------------------------
        # Save Latest Checkpoint
        # ---------------------------------------------

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_acc,
            },
            last_checkpoint
        )

    print("\n" + "=" * 60)
    print("Training Finished")
    print(f"Best Validation Accuracy : {best_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
