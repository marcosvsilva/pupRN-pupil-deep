import cv2
import numpy as np


class ProcessImage:
    def __init__(self):
        # Params

        self._alpha = 1.0
        self._alpha_max = 500
        self._beta = 0
        self._beta_max = 200
        self._gamma = 1.0
        self._gamma_max = 200

        # Brightness Params
        self._range_accept_brightness = 10
        self._max_size_list_color = 10
        self._min_size_list_color_process = 3

        # Variables
        self._mean_color = np.array([])

    def _pre_process(self, frame, gray=False):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (9, 9), 3)
        median = cv2.medianBlur(gaussian, 3)

        kernel = np.ones((5, 5), np.uint8)
        erode = cv2.erode(median, kernel=kernel, iterations=1)
        return cv2.dilate(erode, kernel=kernel, iterations=1)

    def _gamma_corret_image(self, image, gamma):
        look_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        return cv2.LUT(image, look_table)

    def _add_brightness(self, image):
        return cv2.convertScaleAbs(image, alpha=1.0, beta=10)

    def _rem_brightness(self, image):
        return cv2.convertScaleAbs(image, alpha=1.0, beta=-10)

    def _treat_brightness(self, img_process, img_original):
        mean_image = int(self._mean_color.mean())
        range_min, range_max = mean_image - self._range_accept_brightness, mean_image + self._range_accept_brightness
        if int(np.array(img_process).mean()) not in range(range_min, range_max, 1):
            if int(np.array(img_process).mean()) >= range_max:
                new_image = self._rem_brightness(img_original)
            else:
                new_image = self._add_brightness(img_original)

            new_process = self._pre_process(new_image, gray=True)
            img_process = self._treat_brightness(new_image, new_process)

        return img_process

    def process_image(self, img_original):
        if img_original is None:
            return 0

        img_process = self._pre_process(img_original)

        if len(self._mean_color) < self._max_size_list_color:
            self._mean_color = np.append(self._mean_color, np.array(img_process).mean())

        if len(self._mean_color) >= self._min_size_list_color_process:
            img_process = self._treat_brightness(img_process, img_original)

        return img_process
