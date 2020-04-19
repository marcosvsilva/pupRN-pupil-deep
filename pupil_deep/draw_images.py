import cv2


class DrawImages:
    def __init__(self):
        # Params color
        self._white_color = (255, 255, 0)
        self._gray_color = (170, 170, 0)

        # Params text and circle print image
        self._position_text = (30, 30)
        self._font_text = cv2.FONT_HERSHEY_DUPLEX
        self._size_point_pupil = 5

    def mark_eye(self, image, right, left):
        cv2.line(image, (right[0], right[1]), (left[0], left[1]), self._white_color, 1)
        return image

    def mark_center(self, image, center):
        color = self._white_color
        cv2.line(image, (center[0] - 10, center[1]), (center[0] + 10, center[1]), color, 1)
        cv2.line(image, (center[0], center[1] - 10), (center[0], center[1] + 10), color, 1)
        return image

    def draw_circles(self, image, points, radius=0, color=None):
        for point in points:
            rad = radius if radius > 0 else self._size_point_pupil
            paint = self._gray_color if color is None else color
            cv2.circle(image, (point[0], point[1]), rad, paint, 2)

        return image
