import numpy as np
from utils.sort import Sort

class Tracker:
    def __init__(self):
        # max_age: Maximum number of frames to keep alive a track without associated detections.
        # min_hits: Minimum number of associated detections before track is initialised.
        # iou_threshold: Minimum IOU for match.
        self.tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)
        
    def update(self, detections):
        """
        Takes bounding boxes from detector and assigns reliable IDs via SORT.
        detections: list of [x1, y1, x2, y2, score]
        Returns updated tracked objects: array of [x1, y1, x2, y2, id] 
        where id is persistent across frames.
        """
        if len(detections) == 0:
            return self.tracker.update(np.empty((0, 5)))
        
        return self.tracker.update(np.array(detections))
