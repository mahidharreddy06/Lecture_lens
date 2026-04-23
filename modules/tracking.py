import numpy as np
from utils.sort import Sort

class Tracker:
    def __init__(self):
        self.tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.05)
        
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
