import numpy as np
from pupil_deep import PupilDeep

class Pupil:
    def __init__(self):
        self._orientations = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']

        self._image = []
        self._default_color = 0
        self._range_eye = 0
        self._center = []
        self._shape = []

        self._white_range = range(200, 255)

        self._pupul_deep = PupilDeep()

    def pupil_detect(self, image):
        self._image = image

        self._center = self._pupul_deep.run(image)

        self._default_color = image[self._center[0], self._center[1]]

        self._range_eye = range((self._default_color-int(255*0.05)), (self._default_color+int(255*0.05)))

        self._shape = image.shape

        points, radius = self._radius()

        return self._center, radius

    def _radius(self):
        i, j = self._center
        radius = np.array([])
        points = np.array([])
        for orientation in self._orientations:
            point = np.array(self._search_edge(orientation))
            points = np.append(points, point)
            radius = np.append(radius, self._calc_radius(point))

        return points, points.mean()

    def _calc_radius(self, points):
        if self._center[0] != points[0]:
            return abs(self._center[0] - points[0])
        else:
            return abs(self._center[1] - points[1])

    def _search_edge(self, orientation):
        i, j = self._center
        while 0 <= i < self._shape[0] and 0 <= j < self._shape[1]:
            if (self._image[i, j] not in self._range_eye) and (self._image[i, j] not in self._white_range):
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
            i -= 1
        elif orientation == 'north':
            i += 1
        elif orientation == 'east':
            j += 1
        elif orientation == 'west':
            j -= 1
        return i, j
