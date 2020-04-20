import cv2
import numpy as np
import matplotlib.pyplot as pl

from process_image import ProcessImage
from pupil import Pupil
from pupil_deep import PupilDeep
from draw_images import DrawImages
from information import Information


class Execution:
    def __init__(self):
        # Paths
        self._path_label = ''

        # Stops and executions
        self._frame_start = 0
        self._frame_stop = 0

        # Execution params
        self._execute_invisible = True
        self._generate_histogram = False

        # Params color
        self._white_color = (255, 255, 0)
        self._gray_color = (170, 170, 0)

        # Params text and circle print image
        self._position_text = (30, 30)
        self._font_text = cv2.FONT_HERSHEY_DUPLEX
        self._size_point_pupil = 5

        # Params dataset labels out
        self._title_label = 'patient,param,frame,center_x,center_y,radius,flash_algorithm,flash_information,' \
                            'color_flash,eye_size,img_mean,img_std,img_median'

    def _add_label(self, information):
        with open(self._path_label, 'a', newline='') as file:
            file.write('{}\n'.format(information))
            file.close()

    def _add_params_label(self, params):
        number_params = len(self._title_label.split(','))
        if len(params) == number_params:
            params = [str(x) for x in params]
            information = ','.join(params)
            self._add_label(information)
        else:
            raise Exception("Number params execute is difference of number params label!")

    def _show_image(self, image, label, number_frame, path_out, color=None):
        paint = self._white_color if color is None else color
        cv2.putText(image, label, self._position_text, self._font_text, 0.9, paint)

        if not self._execute_invisible:
            cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
            cv2.imshow('Analysis', image)
            cv2.waitKey(1)

        self._save_images({'final': image}, number_frame, path_out)

    def _save_histogram(self, histogram, number_frame, path_out):
        # TODO: Don't work
        if self._generate_histogram:
            pl.hist(histogram, bins='auto')
            pl.title('Histogram Frame: {}'.format(number_frame))
            pl.xlabel("Value")
            pl.ylabel("Frequency")
            pl.savefig("{}/histogram_{}.png".format(path_out, number_frame))

    def _save_images(self, images, number_frame, path_out, center=(0, 0)):
        for key, value in images.items():
            if 'binary' in key:
                draw = DrawImages()
                image = draw.mark_center(value, center)
            else:
                image = value

            out = '{}/{}_{}.png'.format(path_out, key, number_frame)
            cv2.imwrite(out, image)

    def pupil_process(self, paths):
        pupil_deep = PupilDeep()
        process = ProcessImage()
        pupil = Pupil(pupil_deep)
        draw = DrawImages()
        information = Information()

        self._path_label = paths['path_label']
        self._add_label(self._title_label)

        exam = cv2.VideoCapture(paths['path_exam'])
        fps = exam.get(cv2.CAP_PROP_FPS)

        patient_exam, param_exam = information.get_information_exam(paths['path_information'], fps)

        number_frame = 0
        while True:
            _, frame = exam.read()

            if (frame is None) or ((self._frame_stop > 0) and (number_frame >= self._frame_stop)):
                break

            if (self._frame_start > 0) and (number_frame < self._frame_start):
                number_frame += 1
                continue

            original = np.copy(frame)
            img_orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            self._save_images({'original': original}, number_frame, paths['path_out'])

            img_process, flash = process.process_image(original)
            self._save_images({'process': img_process}, number_frame, paths['path_out'])

            img_mean, img_std, img_median = img_process.mean(), img_process.std(), np.median(img_process)

            center, radius, points, images, mean_binary = pupil.pupil_detect(img_process)

            binary = images['binary_pre_process']
            binary = draw.mark_center(binary, center)
            binary = draw.draw_circles(binary, points, 2, self._white_color)
            self._save_images({'binary': binary}, number_frame, paths['path_out'], center)

            img_process = draw.mark_center(img_process, center)
            img_process = draw.draw_circles(img_process, points, 2, self._white_color)
            img_process = draw.draw_circles(img_process, [(center[0], center[1])], radius, self._white_color)
            self._save_images({'img_process': img_process}, number_frame, paths['path_out'])

            self._save_histogram(images['histogram'], number_frame, paths['path_out'])

            img_presentation = cv2.hconcat([img_orig_gray, binary, img_process])

            label = 'Frame=%d;Radius=%d;Center=(%d,%d);BinMean=(%f)' % (
                number_frame, radius, center[0], center[1], mean_binary)

            self._show_image(img_presentation, label, number_frame, paths['path_out'])

            flash_information, color_information = information.get_information_params(number_frame)

            params = [patient_exam, param_exam, number_frame, center[0], center[1], radius, flash, flash_information,
                      color_information, 0, img_mean, img_std, img_median]

            self._add_params_label(params)

            number_frame += 1

        cv2.destroyAllWindows()
        exam.release()
