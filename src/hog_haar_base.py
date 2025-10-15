import cv2
import numpy as np


class HogHaarDetector:
    def __init__(self, use_haar=False, haar_xml_path="haarcascade_upperbody.xml"):
        self.use_haar = use_haar

        if use_haar:
            full_path = cv2.data.haarcascades + haar_xml_path
            self.detector = cv2.CascadeClassifier(full_path)
            if self.detector.empty():
                raise ValueError(f"Could not load Haar cascade from: {full_path}")
            print(f"Loaded Haar cascade: {haar_xml_path}")
        else:
            self.detector = cv2.HOGDescriptor()
            self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("Loaded HOG descriptor with default people detector")

    def _apply_nms(self, boxes, overlap_threshold=0.3):
        if len(boxes) == 0:
            return []

        boxes_array = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        x1, y1, x2, y2 = boxes_array[:, 0], boxes_array[:, 1], boxes_array[:, 2], boxes_array[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.argsort(y2)
        picked = []

        while len(indices) > 0:
            last = len(indices) - 1
            i = indices[last]
            picked.append(i)
            suppress = [last]

            for pos in range(last):
                j = indices[pos]
                xx1, yy1 = max(x1[i], x1[j]), max(y1[i], y1[j])
                xx2, yy2 = min(x2[i], x2[j]), min(y2[i], y2[j])
                w, h = max(0, xx2 - xx1 + 1), max(0, yy2 - yy1 + 1)
                overlap = float(w * h) / areas[j]
                if overlap > overlap_threshold:
                    suppress.append(pos)
            indices = np.delete(indices, suppress)

        return [boxes[i] for i in picked]

    def draw_boxes(self, img, boxes, color=(0, 255, 0), thickness=2):
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            label = f"Person {i + 1}"
            text_y = max(y - 5, 20)
            cv2.putText(img, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, thickness)

        count_text = f"Detections: {len(boxes)}"
        cv2.putText(img, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        return img
