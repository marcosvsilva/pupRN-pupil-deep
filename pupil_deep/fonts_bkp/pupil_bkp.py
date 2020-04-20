import numpy as np
import cv2
from pupil_deep import PupilDeep


class Pupil:
    def __init__(self):
        # Orientations
        self._orientations = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']

        # Variables
        self._center = []
        self._default_color = 0
        self._number_of_points_variable_radius = 2

        # Dependences
        self._pupil_deep = PupilDeep()

        # Params
        self._threshold_min = 25
        self._threshold_max = 255
        self._radius_range = range(35, 100, 1)
        self._pupil_color_range = range(0, 70, 1)
        self._new_color_pupil = 10

        # New Params
        self._radius_search_pupil = 80
        # self._radius_search_pupil = 100
        # self._radius_search_pupil = 60

        self._distance_validate_pupil = 20
        # self._distance_validate_pupil = 10

        self._initial_ranges = {}
        self._execution_ranges = {}

    def _search_pupil(self, binary):
        self._initialize_range(binary)

        rows, columns = binary.shape

        for x in range(self._initial_ranges['top'], self._initial_ranges['bottom']):
            for y in range(self._initial_ranges['left'], self._initial_ranges['right']):
                if y > self._initial_ranges['left'] and (binary[y, x] != 0):
                    if (x + self._distance_validate_pupil <= rows) and (x - self._distance_validate_pupil > 0):
                        if binary[y, x + self._distance_validate_pupil] != 0:
                            self._capture_measures(x, y)
                        if binary[y, x - self._distance_validate_pupil] != 0:
                            self._capture_measures(x, y)

        self._calc_pupil_area()

    def _initialize_range(self, binary):
        i, j = self._center
        top, bottom = i - self._radius_search_pupil, i + self._radius_search_pupil
        left, right = j - self._radius_search_pupil, j + self._radius_search_pupil

        if top < 0:
            top = 0

        if bottom > binary.shape[1]:
            bottom = binary.shape[1]

        if left < self._distance_validate_pupil:
            left = self._distance_validate_pupil

        if right + self._distance_validate_pupil > binary.shape[0]:
            right = binary.shape[0] - self._distance_validate_pupil

        self._initial_ranges = {'top': top, 'bottom': bottom, 'left': left, 'right': right}

        self._execution_ranges = {'top': bottom, 'bottom': top, 'left': right, 'right': left}

    def _capture_measures(self, x, y):
        self._execution_ranges['top'] = x if self._execution_ranges['top'] > x else self._execution_ranges['top']
        self._execution_ranges['left'] = y if self._execution_ranges['left'] > y else self._execution_ranges['left']
        self._execution_ranges['right'] = y if self._execution_ranges['right'] < y else self._execution_ranges['right']
        self._execution_ranges['bottom'] = x if self._execution_ranges['bottom'] < x else self._execution_ranges['bottom']

    def _calc_pupil_area(self):
        i = int(((self._execution_ranges['bottom'] - self._execution_ranges['top']) / 2) + self._execution_ranges['top'])
        j = int(((self._execution_ranges['right'] - self._execution_ranges['left']) / 2) + self._execution_ranges['left'])

        self._radius = int((self._execution_ranges['right'] - self._execution_ranges['left']) / 2)
        self._center = (i, j)

    def _binarize(self, image, threshold_min=0, threshold_max=0, typ=None):
        t_type = cv2.THRESH_BINARY if typ is None else typ
        t_min = self._threshold_min if threshold_min <= 0 else threshold_min
        t_max = self._threshold_max if threshold_max <= 0 else threshold_max
        return cv2.threshold(image, t_min, t_max, t_type)[1]

    def pupil_detect(self, image):
        self._center = self._pupil_deep.run(image)

        hist = np.histogram(image)

        binary_pre_process = self._binarize(image, 10, 255, cv2.THRESH_BINARY_INV)

        self._search_pupil(binary_pre_process)

        images = {'binary_pre_process': binary_pre_process, 'histogram': hist}

        points = [(self._initial_ranges['top'], self._initial_ranges['left']),
                  (self._initial_ranges['top'], self._initial_ranges['right']),
                  (self._initial_ranges['bottom'], self._initial_ranges['left']),
                  (self._initial_ranges['bottom'], self._initial_ranges['right']),
                  (self._execution_ranges['top'], self._execution_ranges['left']),
                  (self._execution_ranges['top'], self._execution_ranges['right']),
                  (self._execution_ranges['bottom'], self._execution_ranges['left']),
                  (self._execution_ranges['bottom'], self._execution_ranges['right'])]

        return self._center, self._radius, points, images
