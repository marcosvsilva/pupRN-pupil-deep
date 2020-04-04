import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as pl
from process_image import ProcessImage
from pupil import Pupil
from eye import Eye


class Main:
    def __init__(self):
        # Global variables for executions
        self._title_exam = ''
        self._path_dataset_out = ''

        # Dependences
        self._process = ProcessImage()
        self._pupil = Pupil()
        self._eye = Eye()

        # Limit cash dependences
        self._max_execution_with_cash = 20

        # Directoris
        self._projects_path = '/media/marcos/Dados/Projects'

        self._dataset_path = '{}/Datasets/exams'.format(self._projects_path)
        self._dataset_out = '{}/Results/PupilDeep/Frames'.format(self._projects_path)
        self._dataset_label = '{}/Results/PupilDeep/Labels'.format(self._projects_path)

        # Stops and executions
        self._frame_stop = 0
        self._frame_start = 0

        self._movie_stop = 0
        self._list_not_available = []
        # self._list_available = ['25080225_08_2019_08_37_59', '25080225_08_2019_08_40_12', '25080425_08_2019_08_53_48', '25080425_08_2019_09_08_25']
        self._list_available = ['benchmark']

        # Params color
        self._white_color = (255, 255, 0)
        self._gray_color = (170, 170, 0)

        # Params text and circle print image
        self._position_text = (30, 30)
        self._font_text = cv2.FONT_HERSHEY_DUPLEX
        self._size_point_pupil = 5

        # Params dataset labels out
        self._title_label = 'frame,center_x,center_y,radius,flash,eye_size,img_mean,img_std,img_median'

    def _clear_cash(self):
        print('clear_cash')
        # # self._process = None
        # del self._pupil
        # del self._eye
        #
        # # self._process = ProcessImage()
        # self._pupil = Pupil()
        # self._eye = Eye()

    def _add_label(self, information):
        with open('{}/{}_label.csv'.format(self._dataset_label, self._title_exam), 'a', newline='') as file:
            file.write('{}\n'.format(information))
            file.close()

    def _make_path(self, path=''):
        try:
            if path == '':
                os.mkdir(self._path_dataset_out)
            else:
                os.mkdir(path)
        except FileExistsError:
            pass

    def _show_image(self, image, label, number_frame, color=None):
        system_continue = True
        paint = self._white_color if color is None else color
        cv2.putText(image, label, self._position_text, self._font_text, 0.9, paint)

        # cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
        # cv2.imshow('Analysis', image)
        # order = cv2.waitKey(1)
        #
        # if order == 32:
        #     time.sleep(2)
        # elif order == ord('q'):
        #     system_continue = False
        
        self._save_images({'final': image}, number_frame)
        return system_continue

    def _save_images(self, images, number_frame, center=(0, 0)):
        for key, value in images.items():
            if 'binary' in key:
                image = self._mark_center(value, center)
            else:
                image = value

            out = '{}/{}_{}.png'.format(self._path_dataset_out, key, number_frame)
            cv2.imwrite(out, image)

    def _save_histogram(self, histogram, number_frame):
        pl.hist(histogram, bins='auto')
        pl.title('Histogram Frame: {}'.format(number_frame))
        pl.xlabel("Value")
        pl.ylabel("Frequency")
        pl.savefig("{}/histogram_{}.png".format(self._path_dataset_out, number_frame))

    def _mark_eye(self, image, right, left):
        cv2.line(image, (right[0], right[1]), (left[0], left[1]), self._white_color, 1)
        return image

    def _mark_center(self, image, center):
        color = self._white_color
        cv2.line(image, (center[0] - 10, center[1]), (center[0] + 10, center[1]), color, 1)
        cv2.line(image, (center[0], center[1] - 10), (center[0], center[1] + 10), color, 1)
        return image

    def _draw_circles(self, image, points, radius=0, color=None):
        for point in points:
            rad = radius if radius > 0 else self._size_point_pupil
            paint = self._gray_color if color is None else color
            cv2.circle(image, (point[0], point[1]), rad, paint, 2)

        return image

    def _pupil_process(self, exam):
        number_frame = 0
        system_continue = True

        while system_continue:
            _, frame = exam.read()

            if (frame is None) or ((self._frame_stop > 0) and (number_frame >= self._frame_stop)):
                break

            if (self._frame_start > 0) and (number_frame < self._frame_start):
                number_frame += 1
                continue

            if (number_frame > 10) and (number_frame % self._max_execution_with_cash) == 0:
                self._clear_cash()

            original = np.copy(frame)
            img_orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            self._save_images({'original': original}, number_frame)

            img_process, flash = self._process.process_image(original)
            self._save_images({'process': img_process}, number_frame)

            img_mean, img_std, img_median = img_process.mean(), img_process.std(), np.median(img_process)

            center, radius, points, images = self._pupil.pupil_detect(img_process)

            binary = images['binary_pre_process']
            binary = self._mark_center(binary, center)
            binary = self._draw_circles(binary, points, 2, self._white_color)
            self._save_images({'binary': binary}, number_frame, center)

            img_process = self._mark_center(img_process, center)
            img_process = self._draw_circles(img_process, points, 2, self._white_color)
            img_process = self._draw_circles(img_process, [(center[0], center[1])], radius, self._white_color)
            self._save_images({'img_process': img_process}, number_frame)

            self._save_histogram(images['histogram'], number_frame)

            img_presentation = cv2.hconcat([img_orig_gray, binary, img_process])
            label = 'Frame=%d;Radius=%d;Center=(%d,%d);Eye=(%d);Flash=(%d)' % (number_frame, radius, center[0],
                                                                               center[1], 0, flash)

            system_continue = self._show_image(img_presentation, label, number_frame)

            self._add_label("{},{},{},{},{},{},{},{}".format(number_frame, center[0], center[1], radius, flash, 0,
                                                             img_mean, img_std, img_median))

            number_frame += 1

        cv2.destroyAllWindows()
        exam.release()

    def run(self):
        start_time = time.time()

        files = os.listdir(self._dataset_path)

        number_movie = 0

        for file in files:
            if (self._movie_stop > 0) and (number_movie >= self._movie_stop):
                break

            self._title_exam = file.replace('.mp4', '')
            self._path_dataset_out = '{}/{}'.format(self._dataset_out, self._title_exam)

            if (len(self._list_available) > 0) and (self._title_exam not in self._list_available):
                continue

            if self._title_exam in self._list_not_available:
                continue

            self._add_label(self._title_label)
            self._make_path()

            # self._clear_cash()
            exam = cv2.VideoCapture('{}/{}'.format(self._dataset_path, file))
            self._pupil_process(exam)

            number_movie += 1

        end_time = time.time()
        self._add_label('Execition time: {} minuts'.format((end_time - start_time) / 60))


main = Main()
if __name__ == '__main__':
    main.run()
