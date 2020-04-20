import cv2
import numpy as np


class ProcessImage:
    def __init__(self):
        # Flash params
        self._length_list_color = 10
        self._range_color = 100
        self._median_color = 0

        # Params process image
        self._alpha = 1.0
        self._alpha_max = 500
        self._beta = 0
        self._beta_max = 200
        self._gamma = 1.0
        self._gamma_max = 200

        # Variables
        self._list_color_images = np.array([])

    def _pre_process(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (9, 9), 3)
        median = cv2.medianBlur(gaussian, 3)

        kernel = np.ones((5, 5), np.uint8)
        erode = cv2.erode(median, kernel=kernel, iterations=1)
        return cv2.dilate(erode, kernel=kernel, iterations=1)

    def process_image(self, img_original):
        reflex = False
        if img_original is None:
            return 0

        if len(self._list_color_images) >= self._length_list_color:
            normal_range = range(self._median_color - self._range_color, self._median_color + self._range_color, 1)
            reflex = int(img_original.mean()) not in normal_range
        else:
            self._list_color_images = np.append(self._list_color_images, np.array(img_original).mean())
            self._median_color = int(self._list_color_images.mean())

        img_process = self._pre_process(img_original)

        return img_process, int(reflex)
