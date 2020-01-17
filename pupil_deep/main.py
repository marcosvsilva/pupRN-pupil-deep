import cv2
import os
import numpy as np
from pupil import Pupil


class Main:
    def __init__(self):
        # Params
        self._frame_stop = 20
        self._movie_stop = 1

        self.dataset_path = 'eye_test/movies'
        self.dataset_out = 'eye_test/out'
        self.dataset_label = 'eye_test/label'

        self._white_color = (255, 255, 0)
        self._gray_color = (170, 170, 0)
        self._black_color = (0, 0, 0)
        self._black_color_range = range(0, 150, 1)

        self._size_point_pupil = 5

        self._position_text = (30, 30)
        self._font_text = cv2.FONT_HERSHEY_DUPLEX

        self._title = ''
        self._dataset_out_exam = ''

        self._pupil = Pupil()

    def _add_label(self, information):
        with open('{}/{}_label.csv'.format(self.dataset_label, self._title), 'a', newline='') as file:
            file.write('{}\n'.format(information))
            file.close()

    def _make_path(self, path=''):
        try:
            if path == '':
                os.mkdir(self._dataset_out_exam)
            else:
                os.mkdir(path)
        except FileExistsError:
            pass

    def _show_image(self, image, label, number_frame):
        cv2.putText(image, label, self._position_text, self._font_text, 0.9, self._white_color)

        cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
        cv2.imshow('Analysis', image)
        cv2.waitKey(1)
        self._save_image(image, number_frame)

    def _save_image(self, image, number_frame, title=''):
        if title == '':
            out = '{}/{}.png'.format(self._dataset_out_exam, number_frame)
        else:
            out = '{}/{}_{}.png'.format(self._dataset_out_exam, title, number_frame)
        cv2.imwrite(out, image)

    @staticmethod
    def _pre_process(frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (9, 9), 3)
        median = cv2.medianBlur(gaussian, 3)

        kernel = np.ones((5, 5), np.uint8)
        erode = cv2.erode(median, kernel=kernel, iterations=1)
        return cv2.dilate(erode, kernel=kernel, iterations=1)

    def _mark_center(self, image, center):
        #color = self._black_color if image[center[0], center[1]] in self._black_color_range else self._white_color
        color = self._white_color
        cv2.line(image, (center[0] - 10, center[1]), (center[0] + 10, center[1]), color, 1)
        cv2.line(image, (center[0], center[1] - 10), (center[0], center[1] + 10), color, 1)
        return image

    def _draw_circles(self, image, points):
        for point in points:
            cv2.circle(image, (point[0], point[1]), self._size_point_pupil, self._gray_color, 2)

        return image

    def _pupil_process(self, exam):
        number_frame = 0

        while True:
            _, frame = exam.read()

            if (frame is None) or (number_frame >= self._frame_stop):
                break

            original = np.copy(frame)

            img_process = self._pre_process(original)

            center, radius, points, binary = self._pupil.pupil_detect(img_process)
            binary = self._mark_center(binary, center)
            self._save_image(binary, number_frame, 'binary')

            img_process = self._mark_center(img_process, center)
            img_process = self._draw_circles(img_process, points)
            cv2.circle(img_process, (center[0], center[1]), radius, self._white_color, 3)

            label = 'Frame=%d;Radius=%d;Center=(%d,%d)' % (number_frame, radius, center[0], center[1])
            self._show_image(img_process, label, number_frame)

            self._add_label("{},{},{}".format(number_frame, center[0], center[1]))

            number_frame += 1

        exam.release()
        cv2.destroyAllWindows

    def run(self):
        files = os.listdir(self.dataset_path)

        number_movie = 0

        for file in files:
            if number_movie >= self._movie_stop:
                pass

            self._title = file.replace('.mp4', '')
            self._dataset_out_exam = '{}/{}'.format(self.dataset_out, self._title)

            if self._title != '07080407_08_2019_09_33_39':
                pass

            self._add_label('frame,x,y')
            self._make_path()

            exam = cv2.VideoCapture('{}/{}'.format(self.dataset_path, file))
            self._pupil_process(exam)

            number_movie += 1


main = Main()
if __name__ == '__main__':
    main.run()
