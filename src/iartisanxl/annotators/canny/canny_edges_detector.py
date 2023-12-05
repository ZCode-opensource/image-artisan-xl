import numpy as np
import cv2


class CannyEdgesDetector:
    # pylint: disable=no-member
    def get_canny_edges(self, image, low_threshold: int, high_threshold: int, resolution=None):
        original_resolution = (image.shape[0], image.shape[1])

        if resolution:
            image = cv2.resize(image, (resolution[1], resolution[0]))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)
        edges = cv2.resize(edges, (original_resolution[1], original_resolution[0]))
        edges = np.stack([edges] * 3, axis=-1)

        return edges
