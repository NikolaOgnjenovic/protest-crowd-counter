import cv2
from pathlib import Path
from hog_haar_base import HogHaarDetector


def run_haar(image_path, save_path="../data/haar_result.jpg", haar_xml="haarcascade_upperbody.xml"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    print(f"Running Haar detector with {haar_xml}")
    haar_detector = HogHaarDetector(use_haar=True, haar_xml_path=haar_xml)
    boxes = haar_detector.detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    boxes = haar_detector._apply_nms(boxes)
    img_out = haar_detector.draw_boxes(img.copy(), boxes, color=(255, 0, 0))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img_out)
    print(f"Haar: Detected {len(boxes)} people, saved to {save_path}")
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
        haar_xml = "haarcascade_upperbody.xml"
        if scenario == "closeup":
            haar_xml = "haarcascade_frontalface_default.xml"
        elif scenario == "pedestrians":
            haar_xml = "haarcascade_fullbody.xml"

        print(f"\nTesting Haar on {scenario} scenario: {img_path}")
        run_haar(img_path, f"../data/{scenario}_haar.jpg", haar_xml=haar_xml)


if __name__ == "__main__":
    main()
