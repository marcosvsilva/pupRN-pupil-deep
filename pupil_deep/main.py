import cv2
import os
import numpy as np
from pupil import Pupil


class Main:
    def __init__(self):
        # Params
        self.dataset_path = 'eye_test/movies'
        self.dataset_out = 'eye_test/out'
        self.dataset_label = 'eye_test/label'

        self._pupil_color = (255, 255, 0)
        self._point_color = (170, 170, 0)
        self._font_color = (255, 255, 0)

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
        cv2.putText(image, label, self._position_text, self._font_text, 0.9, self._font_color)

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

    def _pupil_process(self, exam):
        number_frame = 0

        while True:
            _, frame = exam.read()

            if frame is None:
                break

            original = np.copy(frame)

            img_process = self._pre_process(original)

            center, radius, points, binary = self._pupil.pupil_detect(img_process)
            self._save_image(binary, number_frame, 'binary')

            for point in points:
                cv2.circle(original, (point[0], point[1]), self._size_point_pupil, self._point_color, 2)

            cv2.circle(original, (center[0], center[1]), radius, self._pupil_color, 3)

            label = 'Frame=%d;Radius=%d;Center=(%d,%d)' % (number_frame, radius, center[0], center[1])
            self._show_image(original, label, number_frame)

            self._add_label("{},{},{}".format(number_frame, center[0], center[1]))

            number_frame += 1

        exam.release()
        cv2.destroyAllWindows

    def run(self):
        files = os.listdir(self.dataset_path)
        for file in files:
            self._title = file.replace('.mp4', '')
            self._dataset_out_exam = '{}/{}'.format(self.dataset_out, self._title)

            self._add_label('frame,x,y')
            self._make_path()

            exam = cv2.VideoCapture('{}/{}'.format(self.dataset_path, file))
            self._pupil_process(exam)


main = Main()
if __name__ == '__main__':
    main.run()
