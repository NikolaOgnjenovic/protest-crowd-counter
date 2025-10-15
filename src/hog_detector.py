import cv2
from pathlib import Path
from hog_haar_base import HogHaarDetector


def run_hog(image_path, save_path="../data/hog_result.jpg", confidence=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    print("Running HOG detector")
    hog_detector = HogHaarDetector(use_haar=False)
    detections, weights = hog_detector.detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
        hitThreshold=confidence
    )

    # Filter by confidence
    boxes = [detections[i] for i, w in enumerate(weights) if w > max(confidence, 0.3)] if len(weights) > 0 else detections
    boxes = hog_detector._apply_nms(boxes)

    img_out = hog_detector.draw_boxes(img.copy(), boxes, color=(0, 255, 0))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img_out)
    print(f"HOG: Detected {len(boxes)} people, saved to {save_path}")
    return len(boxes)


def main():
    test_images = {
        "closeup": "../data/samples/closeup.jpeg",
        "pedestrians": "../data/samples/pedestrians.jpg",
        "aerial": "../data/samples/aerial.jpeg"
    }

    for scenario, img_path in test_images.items():
        if not Path(img_path).exists():
            print(f"Warning: {img_path} not found, skipping...")
            continue
        print(f"\nTesting HOG on {scenario} scenario: {img_path}")
        run_hog(img_path, f"../data/output/{scenario}_hog.jpg")


if __name__ == "__main__":
    main()
