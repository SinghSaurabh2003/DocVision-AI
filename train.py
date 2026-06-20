import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import AdamW

from src.config import (
    DATASET_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VAL_SPLIT,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
    SEED,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from src.dataset import RVLCDIPDataset
from src.dataloader import get_dataloaders
from src.model import build_model
from src.trainer import train_one_epoch, validate_one_epoch


def main():

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print(f"Using Device : {device}")
    print("=" * 60)

    # -------------------------------------------------
    # Image Transform
    # -------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
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
    # DataLoaders
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

    model = build_model()

    model = model.to(device)

    # -------------------------------------------------
    # Loss Function
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss()

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
    # Checkpoint Folder
    # -------------------------------------------------

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_acc = 0.0

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    for epoch in range(EPOCHS):

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

        print(f"\nTrain Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")

        current_lr = optimizer.param_groups[0]["lr"]
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
                os.path.join(
                    CHECKPOINT_DIR,
                    BEST_MODEL_NAME
                )
            )

            print("\n✅ Best model saved.")

    print("\n" + "=" * 60)
    print("Training Completed")
    print(f"Best Validation Accuracy : {best_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()