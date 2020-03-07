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
        self._min_brightness = 125
        self._max_brightness = 130

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

    def _gamma_corret_image(self, image, gamma):
        look_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        return cv2.LUT(image, look_table)

    def _add_brightness(self, image):
        return cv2.convertScaleAbs(image, alpha=1.0, beta=10)
        # return self._gamma_corret_image(image, 0.4)

    def _rem_brightness(self, image):
        return cv2.convertScaleAbs(image, alpha=1.0, beta=-10)
        # return self._gamma_corret_image(image, 0.4)

    def _treat_brightness(self, image):
        res = np.copy(image)

        if int(image.mean()) not in range(self._min_brightness, self._max_brightness, 1):
            res = self._gamma_corret_image(image, 3)

        return res

        # img_mean = image.mean()
        # if img_mean < self._min_brightness:
        #     image = self._add_brightness(image)
        #     return self._treat_brightness(image)
        # elif img_mean > self._max_brightness:
        #     image = self._rem_brightness(image)
        #     return self._treat_brightness(image)
        # else:
        #     return image

    def process_image(self, img_original):
        if img_original is None:
            return 0

        aux = 0
        alpha_init = int((self._alpha + aux) * 100)
        while alpha_init <= self._alpha_max:
            correct_alpha = self._on_linear_transform_alpha_trackbar(alpha_init, img_original)
            presentation = cv2.hconcat([img_original, correct_alpha])

            cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
            cv2.imshow('Alpha correct', presentation)
            cv2.waitKey(1)
            out = 'a_{}.png'.format(aux)
            cv2.imwrite(out, presentation)

            aux += 1
            alpha_init = int((self._alpha + aux) * 100)

        aux = 0
        beta_init = int((self._beta + aux) * 100)
        while beta_init <= self._beta_max:
            correct_beta = self._on_linear_transform_beta_trackbar(beta_init, img_original)
            presentation = cv2.hconcat([img_original, correct_beta])

            cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
            cv2.imshow('Beta correct', presentation)
            cv2.waitKey(1)
            out = 'b_{}.png'.format(aux)
            cv2.imwrite(out, presentation)

            aux += 1
            beta_init = int((self._beta + aux) * 100)

        aux = 0
        gamma_init = int((self._gamma + aux) * 100)
        while gamma_init <= self._gamma_max:
            correct_gamma = self._on_gamma_correction_trackbar(gamma_init, img_original)
            presentation = cv2.hconcat([img_original, correct_gamma])

            cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
            cv2.imshow('Beta correct', presentation)
            cv2.waitKey(1)
            out = 'g_{}.png'.format(aux)
            cv2.imwrite(out, presentation)

            aux += 1
            gamma_init = int((self._gamma + aux) * 100)

        cv2.destroyAllWindows()

        return 0
