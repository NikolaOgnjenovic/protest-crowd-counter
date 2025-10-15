import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# Dataset with optional tiling
# -----------------------------
class DroneCrowdDataset(Dataset):
    def __init__(self, base_path, list_file, tile_size=(512, 512), transform=None):
        self.tile_size = tile_size
        self.transform = transform
        self.tiles = []  # list of (seq, frame_id_str, image_path, x0, y0)
        # annotations: dict keyed by (seq, frame_id_str) -> list of (x, y)
        self.annotations = {}

        with open(list_file, "r") as f:
            sequences = [line.strip() for line in f if line.strip()]

        for seq in sequences:
            seq_folder = os.path.join(base_path, "sequences", seq)
            ann_file = os.path.join(base_path, "annotations", f"{seq}.txt")

            # parse annotation file: each line: frame_id,x,y
            with open(ann_file, "r") as fa:
                for line in fa:
                    parts = line.strip().split(",")
                    if len(parts) != 3:
                        continue
                    frame_id_str, x, y = parts
                    key = (seq, frame_id_str.zfill(5))
                    self.annotations.setdefault(key, []).append((int(x), int(y)))

            # list images and create tiles
            for img_file in sorted(os.listdir(seq_folder)):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    frame_id_str = os.path.splitext(img_file)[0]  # e.g., 00029
                    img_path = os.path.join(seq_folder, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    for y0 in range(0, h, tile_size[1]):
                        for x0 in range(0, w, tile_size[0]):
                            self.tiles.append((seq, frame_id_str, img_path, x0, y0))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        seq, frame_id_str, img_path, x0, y0 = self.tiles[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tile_h, tile_w = self.tile_size
        img_tile = img[y0:y0 + tile_h, x0:x0 + tile_w]
        if img_tile.size == 0:
            # pad if tile goes out of bounds
            img_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            img_tile = cv2.resize(img_tile, (tile_w, tile_h))

        density_map = np.zeros((tile_h, tile_w), dtype=np.float32)
        key = (seq, frame_id_str)

        # place point annotations as Gaussians within the tile
        if key in self.annotations:
            # original image tile size before resize
            # compute scale from original cropped tile (width w_crop, height h_crop) to (tile_w, tile_h)
            # since we resized the cropped region to tile_size; it's scale is approx tile_w/w_crop, tile_h/h_crop
            # for simplicity, convert points relative to tile crop and then to resized grid
            w_crop = min(tile_w, img.shape[1] - x0)
            h_crop = min(tile_h, img.shape[0] - y0)
            sx = tile_w / max(1, w_crop)
            sy = tile_h / max(1, h_crop)
            for x, y in self.annotations[key]:
                if x0 <= x < x0 + w_crop and y0 <= y < y0 + h_crop:
                    x_rel = int((x - x0) * sx)
                    y_rel = int((y - y0) * sy)
                    if 0 <= x_rel < tile_w and 0 <= y_rel < tile_h:
                        density_map[y_rel, x_rel] = 1.0
        density_map = cv2.GaussianBlur(density_map, (7, 7), 0)

        if self.transform:
            img_tile = self.transform(img_tile)

        return img_tile, torch.from_numpy(density_map[None, :, :])


# -----------------------------
# CSRNet Model
# -----------------------------
# Backward-compat wrapper for GUI import
class CNNCounter(nn.Module):
    def __init__(self, load_weights=True, target_size=(512, 512)):
        super().__init__()
        self.core = CSRNet(load_weights=load_weights, target_size=target_size)

    def forward(self, x):
        return self.core(x)


class CSRNet(nn.Module):
    def __init__(self, load_weights=True, target_size=(512, 512)):
        super().__init__()
        self.target_size = target_size
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if load_weights else None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:30])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x


# -----------------------------
# Training with logging & dynamic GPU usage
# -----------------------------
def train_csrnet(base_path, list_file, tile_size=(512, 512), batch_size=8, epochs=20, lr=1e-5):
    torch.backends.cudnn.benchmark = True  # optimize kernels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DroneCrowdDataset(base_path, list_file, tile_size, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, prefetch_factor=4)

    model = CSRNet(load_weights=True, target_size=tile_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = GradScaler(device='cuda')

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        start_time = time.time()
        loader_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (imgs, densities) in enumerate(loader_iter, 1):
            batch_start = time.time()
            imgs, densities = imgs.to(device, non_blocking=True), densities.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, densities)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_time = time.time() - batch_start

            loader_iter.set_postfix(loss=f"{loss.item():.4f}",
                                    batch_time=f"{batch_time:.1f}s",
                                    gpu_mem=f"{torch.cuda.memory_reserved() / 1024 ** 3:.1f}GB")

        epoch_time = time.time() - start_time
        print(f"--- Epoch {epoch + 1} finished, Avg Loss: {running_loss / len(loader):.4f}, "
              f"Epoch time: {epoch_time:.1f}s ---")

    torch.save(model.state_dict(), "../data/csrnet_drone_tiles.pth")
    print("Training complete. Model saved as csrnet_drone_tiles.pth")
    return model


# -----------------------------
# Prediction
# -----------------------------
def predict_count(model, image_path, tile_size=(512, 512), return_density_map=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    count = 0.0
    density_map_full = np.zeros((h, w), dtype=np.float32)

    for y0 in range(0, h, tile_size[1]):
        for x0 in range(0, w, tile_size[0]):
            tile = img[y0:y0 + tile_size[1], x0:x0 + tile_size[0]]
            if tile.size == 0:
                continue
            tile_resized = cv2.resize(tile, tile_size)
            tile_tensor = transform(tile_resized).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                density_tile = model(tile_tensor)
                # enforce non-negative density
                density_tile = torch.relu(density_tile)
                density_tile = density_tile.squeeze().cpu().numpy()
                density_tile = np.clip(density_tile, 0, None)
                # resize back to the original tile
                density_tile_resized = cv2.resize(density_tile, (tile.shape[1], tile.shape[0]))
                density_map_full[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = density_tile_resized
                count += float(density_tile_resized.sum())

    if return_density_map:
        return count, density_map_full
    else:
        return count


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    base_path = "../data/VisDrone2020-CC"
    train_list = os.path.join(base_path, "trainlist.txt")

    # Train
    model = train_csrnet(base_path, train_list, tile_size=(512, 512), batch_size=16, epochs=20, lr=1e-5)

    # Test
    test_seq = "00011"
    test_img = os.path.join(base_path, "sequences", test_seq, "00001.jpg")
    count, density_map = predict_count(model, test_img)
    print(f"Predicted count: {count:.1f}")

    # Visualize
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.show()
