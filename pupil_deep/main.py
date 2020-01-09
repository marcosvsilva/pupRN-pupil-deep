import deepeye
import cv2
import os
import numpy as np
import random
from pupil import Pupil


class Main:
    def __init__(self):
        self.dataset_path = 'eye_test/movies'
        self.dataset_out = 'eye_test/out'
        self.dataset_label = 'eye_test/label'

        self._white_range = range(150, 255)
        self._default = 1

        self.title = ''
        self.dataset_out_exam = ''

        self.eye_tracker = deepeye.DeepEye()
        self.pupil = Pupil()

    def _add_label(self, information):
        with open('{}/{}_label.csv'.format(self.dataset_label, self.title), 'a', newline='') as file:
            file.write('{}\n'.format(information))
            file.close()

    def _make_path(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    def _show_image(self, image, label, path_save):
        cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
        cv2.imshow('Analysis', image)
        cv2.waitKey(1)

        #cv2.imwrite(path_save, image)

    def _create_range(self, position, limit):
        start = position - 200 if position > 200 else 0
        end = position + 200 if position + 200 < limit else limit - 1
        return start, end

    def _filter_image(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (9, 9), 3)
        median = cv2.medianBlur(gaussian, 3)

        kernel = np.ones((5, 5), np.uint8)
        erode = cv2.erode(median, kernel=kernel, iterations=1)
        return cv2.dilate(erode, kernel=kernel, iterations=1)

    def _extract_eye(self, img):
        center = self.eye_tracker.run(img)
        self._default = img[center[0], center[1]]

        begin_crop_h, end_crop_h = self._create_range(center[0], img.shape[0])
        begin_crop_v, end_crop_v = self._create_range(center[1], img.shape[1])
        return img[begin_crop_h:end_crop_h, begin_crop_v:end_crop_v]

    def _pupil_process(self, exam):
        number_frame = 0

        while True:
            _, frame = exam.read()

            if frame is None:
                break

            original = np.copy(frame)
            img = self._filter_image(original)

            center = self.eye_tracker.run(img)
            cv2.circle(frame, (center[0], center[1]), 10, (255, 255, 255), 2)

            #presentation = cv2.hconcat([eye_img, filter_eye, threshold])
            self._show_image(frame, '', '')

            #self._add_label("{},{},{}".format(number_frame, center[0], center[1]))

        exam.release()
        cv2.destroyAllWindows

    def run(self):
        files = os.listdir(self.dataset_path)
        for file in files:
            self.title = file.replace('.mp4', '')
            self._add_label('frame,x,y')

            self.dataset_out_exam = '{}/{}'.format(self.dataset_out, self.title)
            self._make_path(self.dataset_out_exam)

            exam = cv2.VideoCapture('{}/{}'.format(self.dataset_path, file))
            self._pupil_process(exam)


main = Main()
if __name__ == '__main__':
    main.run()
