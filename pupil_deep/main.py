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

        self._color_circle = (255, 255, 0)

        self._position_text = (120, 30)
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
        out = '{}/{}.png'.format(self._dataset_out_exam, number_frame)

        cv2.putText(image, label, self._position_text, self._font_text, 1, self._color_circle)

        cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
        cv2.imshow('Analysis', image)
        cv2.waitKey(1)

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

            center, radius = self._pupil.pupil_detect(img_process)

            cv2.circle(original, center, 20, (255, 255, 255), 2)

            # presentation = cv2.hconcat([eye_img, filter_eye, threshold])
            label = 'Frame = %d, Radius = %d, Center = (%d, %d)' % (number_frame, radius, center[0], center[1])
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
