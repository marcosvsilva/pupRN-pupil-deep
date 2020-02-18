import numpy as np
import cv2


class Eye:
    def __init__(self):
        # Variables
        self._center = []
        self._default_color = 0
        self._shape = []

        # Params
        self._thresh_binary = 50
        self._threshold_binary = 255
        self._distance_range = range(50, 200, 1)

    def _binarize(self, image):
        return cv2.threshold(image, self._thresh_binary, self._threshold_binary, cv2.THRESH_BINARY)[1]

    def _eye_edge_up(self, image, position, orientation):
        i, j = position
        # Erro aqui - RecursionError: maximum recursion depth exceeded in comparison
        if (0 <= j < self._shape[0] - 1) and (0 <= i < self._shape[1] - 1):
            if image[j, i] == self._default_color:
                i -= 1 if orientation == 'left' else - 1
                border = self._eye_edge_up(image, [i, j], orientation)
            else:
                v = i + 1 if orientation == 'left' else i - 1
                if image[j + 1, v] == self._default_color:
                    border = self._eye_edge_up(image, [i, j + 1], orientation)

                else:
                    border = [i, j]
        else:
            border = [i, j]

        return border

    def _eye_edge_down(self, image, position, orientation):
        i, j = position
        if (0 <= i < self._shape[0]) and (0 <= j < self._shape[1]):
            if image[j, i] == self._default_color:
                i -= 1 if orientation == 'left' else - 1
                border = self._eye_edge_down(image, [i, j], orientation)
            else:
                v = i + 1 if orientation == 'left' else i - 1
                if image[j - 1, v] == self._default_color:
                    border = self._eye_edge_down(image, [i, j - 1], orientation)
                else:
                    border = [i, j]
        else:
            border = [i, j]

        return border

    def _calc_distance(self, points):
        return int((((self._center[0]-points[0]) ** 2) + ((self._center[1] - points[1]) ** 2)) ** (1/2))

    def eye_detect(self, image, center):
        self._center = center
        self._shape = image.shape

        binary = self._binarize(image)

        self._default_color = binary[center[1], center[0]]

        left_point_up = self._eye_edge_up(binary, self._center, 'left')
        left_point_down = self._eye_edge_down(binary, self._center, 'left')
        right_point_up = self._eye_edge_up(binary, self._center, 'right')
        right_point_down = self._eye_edge_down(binary, self._center, 'right')

        dist_left_up, dist_left_down = self._calc_distance(left_point_up), self._calc_distance(left_point_down)
        dist_right_up, dist_right_down = self._calc_distance(right_point_up), self._calc_distance(right_point_down)

        left = left_point_up if dist_left_up > dist_left_down else left_point_down
        right = right_point_up if dist_right_up > dist_right_down else right_point_down

        return left, right, int(self._calc_distance(left) + self._calc_distance(right)), binary
