import numpy as np
import cv2
import pandas as pd
from pupil_deep import PupilDeep


class Pupil:
    def __init__(self):
        # Orientations
        self._orientations = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
        # self._orientations = ['southeast']

        # Variables
        self._center = []
        self._default_color = 0
<<<<<<< HEAD
        self._number_of_points_variable_radius = 2
=======
        self._number_of_points_variable_radius = 3
>>>>>>> dd3e3c7faaf3ff3d2a6ba49456c11065eecabddb

        # Dependences
        self._pupul_deep = PupilDeep()

        # Params
        self._thresh_binary = 25
        self._threshold_binary = 255
        self._radius_range = range(35, 90, 1)
        self._pupil_color_range = range(0, 70, 1)
        self._new_color_pupil = 10
        self._range_search_reflex = 60

    # def _treatment_reflex(self, image, binary):
    #     new_image = np.copy(image)
    #
    #     i, j = self._center
    #     for x in range(i-self._range_search_reflex, i+self._range_search_reflex):
    #         for y in range(j-self._range_search_reflex, j+self._range_search_reflex):
    #             if (0 <= y < image.shape[0] - 1) and (0 <= x < image.shape[1] - 1):
    #                 if binary[y, x] != 0:
    #                     new_image[y, x] = self._new_color_pupil
    #
    #     return new_image

    def _treatment_reflex(self, image, binary):
        new_image = np.copy(image)

        i, j = self._center
        for x in range(i-self._range_search_reflex, i+self._range_search_reflex):
            for y in range(j-self._range_search_reflex, j+self._range_search_reflex):
                if (0 <= y < image.shape[0] - 1) and (0 <= x < image.shape[1] - 1):
                    if image[y, x] not in self._pupil_color_range:
                        new_image[y, x] = self._new_color_pupil

        return new_image

    def _binarize(self, image):
        return cv2.threshold(image, self._thresh_binary, self._threshold_binary, cv2.THRESH_BINARY)[1]

    def _calc_radius(self, image):
        radius, points = np.array([]), []
        for orientation in self._orientations:
            points.append(self._search_edge(image, orientation))

        return points, self._filter_radius(points)

    def _filter_radius(self, points):
        edges = [{'point': x, 'rad': self._calc_distance(x)} for x in points]
        edges = sorted(edges, key=lambda k: k['rad'])
        edges = edges[1:len(self._orientations)-1:1]
        radius = pd.Series([x['rad'] for x in edges])
        median = radius.median()
        std = radius.std()

        close = []
        for rad in edges:
            dist = abs(rad['rad'] - median)
            if dist < std:
                close.append({'rad': rad['rad'], 'distance': dist})

        close = sorted(close, key=lambda k: k['distance'])
        close = pd.Series(x['rad'] for x in close[0:self._number_of_points_variable_radius:1])
        return close.mean()

    def _calc_distance(self, point):
        return int((((self._center[0]-point[0]) ** 2) + ((self._center[1] - point[1]) ** 2)) ** (1/2))

    def _search_edge(self, image, orientation):
        i, j = self._center
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

    def _inc_coordinates(self, orientation, position):
        i, j = position[0], position[1]
        if orientation == 'south':
            j += 1
        elif orientation == 'north':
            j -= 1
        elif orientation == 'east':
            i += 1
        elif orientation == 'west':
            i -= 1
        return i, j

    def pupil_detect(self, image, number_cal=1):
        original = np.copy(image)

        self._center = self._pupul_deep.run(image)
        binary_pre_process = self._binarize(image)
        image_process = self._treatment_reflex(original, binary_pre_process)

        self._center = self._pupul_deep.run(image_process)
        binary_process = self._binarize(image_process)

        self._default_color = binary_process[self._center[1], self._center[0]]

        points, radius = self._calc_radius(binary_process)

        if int(radius) not in self._radius_range:
            radius = 0

        images = {'original': original, 'binary_pre_process': binary_pre_process, 'image_process': image_process,
                  'binary_process': binary_process}

        return self._center, int(radius), points, images
