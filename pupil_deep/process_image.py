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
        self._max_recursion = 10

        # Brightness Params
        self._range_accept_brightness = 10
        self._max_size_list_color = 10
        self._min_size_list_color_process = 3

        # Variables
        self._mean_color = np.array([])

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

    def _gamma_correction(self, img_original, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(img_original, table)

    def _add_brightness(self, image):
        return self._gamma_correction(image, gamma=1.3)

    def _rem_brightness(self, image):
        return self._gamma_correction(image, gamma=0.7)

    def _treat_brightness(self, img_process, img_original):
        mean_images = int(self._mean_color.mean())
        range_limit = range(mean_images - self._range_accept_brightness, mean_images + self._range_accept_brightness, 1)

        recursion = 0
        while (int(np.array(img_original).mean()) not in range_limit) and (recursion < self._max_recursion):
            if int(np.array(img_original).mean()) >= max(range_limit):
                img_original = self._rem_brightness(img_original)
            else:
                img_original = self._add_brightness(img_original)

            img_process = self._pre_process(img_original)
            recursion += 1

        return img_process

    def process_image(self, img_original):
        if img_original is None:
            return 0

        if len(self._mean_color) < self._max_size_list_color:
            self._mean_color = np.append(self._mean_color, np.array(img_original).mean())

        img_process = self._pre_process(img_original)

        self._recursion = 0
        if len(self._mean_color) >= self._min_size_list_color_process:
            img_process = self._treat_brightness(img_process, img_original)

        return img_process
