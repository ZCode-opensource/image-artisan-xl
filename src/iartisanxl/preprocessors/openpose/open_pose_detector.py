import cv2
import numpy as np
import torch

from iartisanxl.preprocessors.openpose.body import Body
from iartisanxl.preprocessors.openpose.hand import Hand
from iartisanxl.preprocessors.openpose import util


class OpenPoseDetector:
    def __init__(self):
        self.body_estimation = Body("./models/preprocessors/openpose/body_pose_model.pth")
        self.hand_estimation = Hand("./models/preprocessors/openpose//hand_pose_model.pth")

    # pylint: disable=no-member
    def get_open_pose(self, image, resolution=None):
        original_resolution = (image.shape[0], image.shape[1])

        if resolution:
            image = cv2.resize(image, (resolution[1], resolution[0]))

        with torch.no_grad():
            candidate, subset = self.body_estimation(image)

            canvas = np.zeros_like(image)
            canvas = util.draw_bodypose(canvas, candidate, subset)

            # detect hand
            hands_list = util.handDetect(candidate, subset, image)

            all_hand_peaks = []
            for x, y, w, _is_left in hands_list:
                peaks = self.hand_estimation(image[y : y + w, x : x + w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                all_hand_peaks.append(peaks)

            canvas = util.draw_handpose(canvas, all_hand_peaks)

        canvas = cv2.resize(canvas, (original_resolution[1], original_resolution[0]))
        return canvas
