
class Pupil:
    def __init__(self):
        self._orientations = ['south', 'north', 'east', 'west', 'northwest', 'northeast', 'southwest', 'southeast']
        self._default_color = 0
        self._range_eye = 0
        self._center = []
        self._shape = []

    def pupil_detect(self, image, center):
        self._default_color = image[center[0], center[1]]
        self._range_eye = range((self._default_color-int(255*0.05)), (self._default_color+int(255*0.05)))
        self._shape = image.shape
        return self._search_edge(image, center)

    def _search_edge(self, image, center):
        edges_position = []
        for orientation in self._orientations:
            i, j = center
            while 0 <= i <= self._shape[0] and 0 <= j <= self._shape[1]:
                if image[i, j] not in self._range_eye:
                    edges_position.append([i, j])
                    break
                else:
                    i, j = self._calc_coordinates(orientation, (i, j))
        return edges_position

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
