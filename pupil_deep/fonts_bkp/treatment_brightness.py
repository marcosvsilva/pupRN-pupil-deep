import cv2
import numpy as np


class TreatmentBrightness:
    def __init__(self):
        # Params
        self._alpha = 1.0
        self._alpha_max = 500
        self._beta = 0
        self._beta_max = 200
        self._gamma = 1.0
        self._gamma_max = 200

    def _basic_linear_transform(self, alpha, beta, img_original):
        return cv2.convertScaleAbs(img_original, alpha=alpha, beta=beta)

    def _gamma_correction(self, img_original, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(img_original, table)

    def _on_linear_transform_alpha_trackbar(self, val, img):
        alpha = val / 100
        return self._basic_linear_transform(alpha, self._beta, img)

    def _on_linear_transform_beta_trackbar(self, val, img):
        beta = val - 100
        return self._basic_linear_transform(self._alpha, beta, img)

    def _on_gamma_correction_trackbar(self, val, img):
        gamma = val / 100
        return self._gamma_correction(gamma, img)

    def treat_brightness(self, img_original):
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
