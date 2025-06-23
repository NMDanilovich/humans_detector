import torch
from torch import nn
import torchvision
import torchvision.transforms.v2 as v2
from torchvision.ops import nms

import cv2

from tools import nms


class YOLOPipeline(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(YOLOPipeline, self).__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.transforms = v2.Compose([
            v2.ToTensor(),
            v2.Resize((448, 448)),
        ])

    def preprocess(self, img: cv2.Mat) -> torch.Tensor:
        """
        Preprocess input image for YOLO model.
        
        Args:
            img: Input image ndarray or cv2.Mat (H,W,3) or (3,H,W) in range 0-255
            
        Returns:
            Processed image tensor (1,3,H,W) normalized and padded
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        img = img.unsqueeze(0)
        
        return img.to(self.device)

    def inference(self, img: torch.Tensor) -> torch.Tensor:
        """
        Run YOLO model inference.
        
        Args:
            img: Preprocessed image tensor (1,3,H,W)
            
        Returns:
            Raw model output tensor
        """
        with torch.no_grad():
            outputs = self.model(img)
        return outputs
    
    def postprocess(self, outputs: torch.Tensor):
        """
        Postprocess YOLO model outputs.
        
        Args:
            outputs: Raw model outputs
            orig_shape: Original image shape (H,W)
            
        Returns:
            Generator of detections (each detection is a dict with 'boxes', 'scores', 'labels')
        """
        outputs = outputs.squeeze()
        
        if outputs.shape[0] == 0:
            return []
        
        # select one class - person
        clss = outputs[::, 5:]
        indxs = torch.argmax(clss, dim=1)
        person = torch.where(indxs == 0)

        # score filtering
        outputs = outputs[person]
        scores = outputs[::, 4]
        score_thresh = torch.where(scores >= 0.65)

        outputs = outputs[score_thresh]

        bboxes = outputs[::, :4]
        scores = outputs[::, 4]
        clss = outputs[::, 5:]
        indxs = torch.argmax(clss, dim=1)
        
        # non-max suprations
        indices = nms(bboxes, scores, iou_threshold=0.4)

        bboxes = bboxes[indices.to(self.device)] / 448
        scores = scores[indices.to(self.device)]
        indxs = indxs[indices.to(self.device)]
        outputs = zip(bboxes, scores, indxs)
            
        return outputs

    def forward(self, x):
        x = self.preprocess(x)
        x = self.inference(x)
        x = self.postprocess(x)
        return x