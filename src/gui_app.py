import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from hog_detector import run_hog
from haar_detector import run_haar
from cnn_counter import CNNCounter, predict_count
import torch
import numpy as np
import os


class App:
    def __init__(self, root):
        """Initialize the GUI app, load models, and set up widgets."""
        self.root = root
        root.title("People Counting")

        # Canvas to display images
        self.canvas = tk.Canvas(root, width=960, height=540)
        self.canvas.pack()

        # Frame to hold buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()

        # Buttons for different actions
        self.btn_load = tk.Button(self.btn_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side="left")

        self.btn_hog = tk.Button(self.btn_frame, text="HOG Detect", command=self.hog_detect)
        self.btn_hog.pack(side="left")

        self.btn_haar = tk.Button(self.btn_frame, text="Haar Detect", command=self.haar_detect)
        self.btn_haar.pack(side="left")

        self.btn_cnn = tk.Button(self.btn_frame, text="CNN Predict", command=self.cnn_predict)
        self.btn_cnn.pack(side="left")

        # Load CNN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = CNNCounter().to(self.device)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../data")

        # Attempt to load pre-trained CNN weights
        loaded = False
        for weight_name in ["csrnet_drone_tiles.pth", "cnn_counter.pth"]:
            weight_path = os.path.join(data_dir, weight_name)
            try:
                if os.path.exists(weight_path):
                    sd = torch.load(weight_path, map_location=self.device)
                    if isinstance(sd, dict) and any(k.startswith("frontend.") or k.startswith("backend.") for k in sd.keys()):
                        sd = {f"core.{k}": v for k, v in sd.items()}

                    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                        sd = sd["state_dict"]

                    model_sd = self.cnn_model.state_dict()
                    overlap = [k for k in sd.keys() if k in model_sd and getattr(sd[k], 'shape', None) == getattr(model_sd[k], 'shape', None)]

                    self.cnn_model.load_state_dict(sd, strict=False)
                    loaded = len(overlap) > 0
                    if loaded:
                        print(
                            f"Loaded density model weights (matched {len(overlap)}/{len(model_sd)} params): {weight_path}")
                        break
                    else:
                        print(f"Weights found but incompatible with CNNCounter: {weight_path}. Trying next.")
            except Exception as e:
                print(f"Failed loading {weight_path}: {e}")
        if not loaded:
            print("No fine-tuned density weights found. Using default model weights.")
        self.cnn_model.eval()

        self.img_path = None  # Path of the loaded image
        self.img = None  # Tkinter PhotoImage

    def load_image(self):
        """Open a file dialog to load an image and display it on the canvas."""
        self.img_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if self.img_path:
            img = cv2.imread(self.img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (960, 540))
            self.img = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

    def hog_detect(self):
        """Run HOG detection on the loaded image and display results."""
        if self.img_path:
            count = run_hog(self.img_path, save_path="../data/gui_hog_result.jpg")
            img = cv2.imread("../data/gui_hog_result.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (960, 540))
            self.img = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
            messagebox.showinfo("HOG Result", f"Detected count: {count}")

    def haar_detect(self):
        """
        Let the user choose the Haar detection scenario (closeup, pedestrians, aerial),
        run Haar detection, and display results.
        """
        if self.img_path:
            # Prompt user for a Haar scenario
            scenario = simpledialog.askstring("Haar Scenario",
                                              "Enter scenario (closeup / pedestrians / aerial):",
                                              initialvalue="pedestrians")
            if scenario is None:
                return  # User cancelled

            # Choose Haar XML based on the scenario
            haar_xml = "haarcascade_upperbody.xml"
            if scenario.lower() == "closeup":
                haar_xml = "haarcascade_frontalface_default.xml"
            elif scenario.lower() == "pedestrians":
                haar_xml = "haarcascade_fullbody.xml"
            elif scenario.lower() == "aerial":
                haar_xml = "haarcascade_upperbody.xml"

            count = run_haar(self.img_path, save_path="../data/gui_haar_result.jpg", haar_xml=haar_xml)

            img = cv2.imread("../data/gui_haar_result.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (960, 540))
            self.img = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
            messagebox.showinfo("Haar Result", f"Detected count: {count}")

    def cnn_predict(self):
        """Run CNN density prediction on the loaded image and overlay the density map."""
        if self.img_path:
            count, density_map = predict_count(self.cnn_model, self.img_path, return_density_map=True)
            img = cv2.imread(self.img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize density map to match image size
            density_map_resized = cv2.resize(density_map, (img.shape[1], img.shape[0]))

            # Normalize and apply a color map for visualization
            dm = np.maximum(density_map_resized, 0)
            dm = dm / (dm.max() + 1e-6)
            overlay = (dm * 255).astype(np.uint8)
            overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
            combined = cv2.resize(combined, (960, 540))

            self.img = ImageTk.PhotoImage(Image.fromarray(combined))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
            messagebox.showinfo("CNN Result", f"Predicted count: {count:.1f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
