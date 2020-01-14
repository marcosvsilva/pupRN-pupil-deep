import numpy as np
import cv2
from pupil_deep import PupilDeep


class Pupil:
    def __init__(self):
        #params
        self._orientations = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']

        self._pupul_deep = PupilDeep()

        self._default_color = 0
        self._thresh_binary = 30
        self._threshold_binary = 255
        self._radius_range = range(35, 100, 1)
        self._pupil_color_range = range(0, 170, 1)
        self._new_color_pupil = 20
        self._range_search_reflex = 30

    def pupil_detect(self, image):
        original = np.copy(image)

        center = self._pupul_deep.run(image)

        binary = self._binarize(image)

        self._default_color = binary[center[0], center[1]]

        points, radius = self._radius(binary, center)

        if int(radius) not in self._radius_range:
            binary = self._mask_reflex(original, binary, center, center)
            #self.pupil_detect(new_image)

        return center, int(radius), points, binary

    def _mask_reflex(self, image, binary, center, position):
        new_image = np.copy(image)

        if abs(position[0] - center[0]) > self._range_search_reflex:
            return new_image

        if abs(position[1] - center[1]) > self._range_search_reflex:
            return new_image

        i, j = position
        for orientation in self._orientations:
            if binary[j, i] != 0:
                new_image[j, i] = self._new_color_pupil

            new_image = self._mask_reflex(new_image, binary, center, self._calc_coordinates(orientation, position))

    @staticmethod
    def _close_img(image, size_kernel):
        kernel = np.ones((size_kernel, size_kernel), np.uint8)
        erode = cv2.erode(image, kernel=kernel, iterations=1)
        return cv2.dilate(erode, kernel=kernel, iterations=1)

    def _binarize(self, image):
        return cv2.threshold(image, self._thresh_binary, self._threshold_binary, cv2.THRESH_BINARY)[1]

    def _radius(self, image, center):
        radius, points = np.array([]), []
        for orientation in self._orientations:
            point = self._search_edge(image, center, orientation)
            points.append(point)
            radius = np.append(radius, self._calc_radius(center, point))

        radius.sort()
        radius = radius[2:6:1]
        return points, radius.mean()

    @staticmethod
    def _calc_radius(center, points):
        if center[0] != points[0]:
            return abs(center[0] - points[0])
        else:
            return abs(center[1] - points[1])

    def _search_edge(self, image, center, orientation):
        i, j = center
        x, y = image.shape
        while 0 <= i < y and 0 <= j < x:
            if image[j, i] != self._default_color:
                break
            else:
                i, j = self._calc_coordinates(orientation, (i, j))
        return [i, j]

    def _calc_coordinates(self, orientation, position):
        if orientation == 'northwest':
            position = self._inc_coordinates('north', position)
            position = self._inc_coordinates('west', position)
        elif orientation == 'northeast':
            position = self._inc_coordinates('north', position)
            position = self._inc_coordinates('east', position)
        elif orientation == 'southwest':
            position = self._inc_coordinates('south', position)
            position = self._inc_coordinates('west', position)
        elif orientation == 'southeast':
            position = self._inc_coordinates('south', position)
            position = self._inc_coordinates('east', position)
        else:
            position = self._inc_coordinates(orientation, position)
        return position

    @staticmethod
    def _inc_coordinates(orientation, position):
        i, j = position[0], position[1]
        if orientation == 'south':
            i += 1
        elif orientation == 'north':
            i -= 1
        elif orientation == 'east':
            j -= 1
        elif orientation == 'west':
            j += 1
        return i, j
