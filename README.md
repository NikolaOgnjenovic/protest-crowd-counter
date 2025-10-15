# Automatic people counting in aerial photos

Automatically estimate the number of participants from drone images using classical CV and deep learning.

What’s included:
- HOG/Haar baseline detectors (OpenCV)
- CSRNet-style density estimation for counting
- Tkinter GUI to compare methods side-by-side and visualize outputs
- VisDrone-2020CC (lite) dataset integration and training pipelines

## Project Files
- `hog_haar_detector.py` — HOG/Haar implementations and utilities to draw boxes.
- `cnn_counter.py` — CSRNet density model, VisDrone tiling dataset, training and prediction helpers. Includes a CNNCounter wrapper for backward compatibility.
- `gui_app.py` — Tkinter GUI to load an image, run HOG/Haar and CSRNet, and visualize results.
- `VisDrone2020-CC/` — Dataset folder (sequences, annotations, trainlist.txt, testlist.txt).

## Installation

Python 3.9+ recommended.

```bash
pip install opencv-python pillow numpy torch torchvision matplotlib tqdm
```

## Dataset: VisDrone-2020CC (lite)
Expected structure:
- `VisDrone2020-CC/sequences/<SEQ>/<FRAME>.jpg` (e.g., sequences/00014/00029.jpg)
- `VisDrone2020-CC/annotations/<SEQ>.txt` with lines: `frame_id,x,y` (point per person)
- `VisDrone2020-CC/trainlist.txt` and `testlist.txt` listing sequence IDs

Notes:
- The annotation files have three values per row: frame index in the sequence and the (x,y) position of a person in pixels. We parse them as (seq, frame_id, x, y).
- For detection training, points are converted to small square boxes centered at (x,y).

## Training

Density (CSRNet):
```bash
cd src
python -c "from cnn_counter import train_csrnet; model=train_csrnet('VisDrone2020-CC','VisDrone2020-CC/trainlist.txt', tile_size=(512,512), batch_size=8, epochs=20, lr=1e-5)"
```

## Inference and Metrics

- Density counting on one image:
```bash
python - << 'PY'
from cnn_counter import CNNCounter, predict_count
import torch
m=CNNCounter().eval()
try:
    m.load_state_dict(torch.load('cnn_counter.pth', map_location='cpu'))
except Exception:
    pass
c, dm = predict_count(m, 'VisDrone2020-CC/sequences/00014/00029.jpg', return_density_map=True)
print('Predicted count=', c)
PY
```

## GUI
Run the GUI to compare methods:
```bash
python src/gui_app.py
```
In the app:
- Load Image — choose a JPG/PNG.
- HOG Detect — runs classical HOG/haar-based detector and draws green boxes.
- CNN Predict — runs the density model and overlays the density heatmap; shows the predicted count.

## Notes on Methods
- HOG (Histogram of Oriented Gradients): analyzes gradient directions in local patches and uses a linear SVM to detect pedestrian-like shapes.
- Haar Cascades: boosted cascades of Haar-like wavelets for fast detection (Viola–Jones). Fast but less robust than CNNs.

## Extending
- Replace Faster R-CNN with YOLOv5/YOLOv8 or SSD if desired; reuse the DetectionDataset and training loop pattern.
- Adjust box_size in `visdrone_cc.DetectionDataset` if persons are very small/large.
- Tune tile_size and Gaussian kernel for density maps in `cnn_counter.py`.
