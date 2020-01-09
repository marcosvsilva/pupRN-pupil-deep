import deepeye
import cv2
import os
import numpy as np
from pupil import Pupil
from reflections import Reflections


class Main:
    def __init__(self):
        self.dataset_path = r'C:\Projects\Datasets\eye_test\movies'
        self.dataset_out = r'C:\Projects\Datasets\eye_test\out'
        self.dataset_label = r'C:\Projects\Datasets\eye_test\label'

        self.title = ''
        self.dataset_out_exam = ''

        self.eye_tracker = deepeye.DeepEye()
        self.pupil = Pupil()
        self.reflections = Reflections()

    def _add_label(self, information):
        with open(r'{}\{}_label.csv'.format(self.dataset_label, self.title), 'a', newline='') as file:
            file.write('{}\n'.format(information))
            file.close()

    def _make_dir(self, dir):
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

    def _pupil_process(self, exam):
        number_frame = 0

        while True:
            _, frame = exam.read()

            if frame is None:
                break

            original = np.copy(frame[:, :, 0])

            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(gray, (9, 9), 3)
            median = cv2.medianBlur(gaussian, 3)

            kernel = np.ones((5, 5), np.uint8)
            erode = cv2.erode(median, kernel=kernel, iterations=1)
            dilate = cv2.dilate(erode, kernel=kernel, iterations=1)
            threshold = cv2.threshold(dilate, 25, 255, cv2.THRESH_BINARY)[1]

            final = np.copy(dilate)

            center = self.eye_tracker.run(final)
            default = dilate[center[0], center[1]]

            img_filter = []
            for line in dilate:
                img_filter.append([x if x not in range(180, 255) else default for x in line])

            final = np.copy(img_filter)

            edges = self.pupil.pupil_detect(final, center)

            lin, col = gray.shape
            if 0 < center[0] < lin and 0 < center[1] < col:
                cv2.circle(final, (int(center[0]), int(center[1])), 10, (255, 255, 0), 2)

            for i in range(len(edges)-1):
                if 0 < edges[i][0] < lin and 0 < edges[i][1] < col:
                    if 0 < edges[i+1][0] < lin and 0 < edges[i+1][1] < col:
                        cv2.line(final, (edges[i][0], edges[i][1]), (edges[i+1][0], edges[i+1][1]), color=(255, 0, 0))

            cv2.line(final, (edges[len(edges)-1][0], edges[len(edges)-1][1]), (edges[0][0], edges[0][1]), color=(255, 0, 0))

            text = 'frame={}, x={}, y={}'.format(number_frame, center[0], center[1])
            cv2.putText(final, text, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            number_frame += 1
            original_out = r'{}\original_{}.png'.format(self.dataset_out_exam, number_frame)
            final_out = r'{}\final_{}.png'.format(self.dataset_out_exam, number_frame)
            presentation = cv2.hconcat([original, final, threshold])

            cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
            cv2.imshow('Analysis', presentation)
            cv2.waitKey(1)

            cv2.imwrite(original_out, final)
            cv2.imwrite(final_out, final)

            self._add_label("{},{},{}".format(number_frame, center[0], center[1]))

        exam.release()
        cv2.destroyAllWindows

    def run(self):
        files = os.listdir(self.dataset_path)
        for file in files:
            self.title = file.replace('.mp4', '')
            self._add_label('frame,x,y')

            self.dataset_out_exam = r'{}\{}'.format(self.dataset_out, self.title)
            self._make_dir(self.dataset_out_exam)

            exam = cv2.VideoCapture('{}/{}'.format(self.dataset_path, file))
            self._pupil_process(exam)


main = Main()
if __name__ == '__main__':
    main.run()
