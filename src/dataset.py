from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class RVLCDIPDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)

        self.image_paths = list(self.root_dir.glob("*.tif"))

        self.classes = sorted(
            list(
                set(
                    "".join(filter(str.isalpha, p.stem))
                    for p in self.image_paths
                )
            )
        )

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        label_name = "".join(filter(str.isalpha, image_path.stem))

        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label