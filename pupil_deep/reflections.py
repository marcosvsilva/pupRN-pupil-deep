

class Reflections:
    def __init__(self):
        self._max_area = [200, 200]
        self._white_range = range(200, 255)
        self._default = 0

    def filter_reflections(self, image, center):
        self._default = image[center[0], center[1]]
        return self._paint_area(image, center)

    def _paint_area(self, image, center):
        i, j = center
        while i-self._max_area[0] <= i <= i+self._max_area[0]:
            while j-self._max_area[1] <= j <= j+self._max_area[1]:
                if 0 <= i < image.shape[0] and 0 <= j < image.shape[1]:
                    if image[i, j] in self._white_range:
                        image[i, j] = self._default
        return image
