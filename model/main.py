import sys
from pathlib import Path

import cv2

from model import YOLOPipeline
from tools import show_results


if __name__ == "__main__":
    # reed the path argument
    video_path = Path(sys.argv[1])

    yolo = YOLOPipeline()

    cap = cv2.VideoCapture(video_path)

    # determine video properties dynamically
    fps = cap.get(cv2.CAP_PROP_FPS)
    # read first frame to get size
    ret0, frame0 = cap.read()
    if not ret0:
        print("Cannot read video. Exit!")
        sys.exit(1)
    height, width = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("./new_video.avi", fourcc, fps, (width, height))
    # rewind to first frame after probing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("No video. Exit!")
            break

        results = yolo(frame)

        new_frame = show_results(frame, results)

        out.write(new_frame)

    cap.release()
    out.release()
    print("")
