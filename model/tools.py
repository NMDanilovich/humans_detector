import cv2
import torch
from torchvision.ops import nms as torch_nms


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """Pytorch NMS, but translate xywh bboxes format to xyxy format"""
    new_boxes = xywh_to_xyxy(boxes)
    return torch_nms(new_boxes, scores, iou_threshold)


def xywh_to_xyxy(boxes) -> torch.Tensor:
    """
    Convert bounding boxes format from [x, y, w, h] to [x1, y1, x2, y2].
    """
    new_boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes)
    new_boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
    new_boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
    new_boxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
    new_boxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2


    return new_boxes

def show_results(image: cv2.Mat, outputs: list | tuple):
    """
    Draw YOLO detection results on the image and display it.

    Args:
        image (cv2.Mat): Original input image.
        outputs (list | tuple): Tensor of detections with shape [N, 6], where each detection is
            [x1, y1, x2, y2, score, classes].
    Returns:
        cv2.Mat: Image with drawn boxes and labels.
    """
    img = image.copy()

    hight, width = img.shape[:2]
    for output in outputs:
        # Unpack detection
        bbox, score, cls = output

        # Convert to integers for drawing
        x1, y1, x2, y2 = xywh_to_xyxy(bbox)
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * hight), int(y2 * hight)

        # Define box color and draw rectangle
        color = (0, 200, 122)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{cls}:{score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Draw background for text for better readability
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    # # Show the image with detections
    # cv2.imshow("YOLO Results", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img
