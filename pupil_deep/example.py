import deepeye
import cv2
import os
import numpy as np


class Main:
    def __init__(self):
        self.dataset_path = r'C:\Projects\Datasets\eye_test\movies'
        self.dataset_out = r'C:\Projects\Datasets\eye_test\out'
        self.dataset_label = r'C:\Projects\Datasets\eye_test\label'

        self.title = ''
        self.dataset_out_exam = ''

        self.eye_tracker = deepeye.DeepEye()

    def add_label(self, information):
        with open(r'{}\{}_label.csv'.format(self.dataset_label, self.title), 'a', newline='') as file:
            file.write('{}\n'.format(information))
            file.close()

    def pupil_process(self, exam):
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

            # tf.placeholder(tf.float32, [None, 28, 28, 1])

            position = self.eye_tracker.run(original)

            lin, col = gray.shape
            if 0 < position[0] < lin and 0 < position[1] < col:
                cv2.circle(original, (int(position[0]), int(position[1])), 10, (255, 255, 0), 2)

            text = 'frame={}, x={}, y={}'.format(number_frame, position[0], position[1])
            cv2.putText(original, text, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            number_frame += 1
            path_out = r'{}\{}.png'.format(self.dataset_out_exam, number_frame)

            cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
            cv2.imshow('Analysis', original)
            cv2.waitKey(1)
            cv2.imwrite(path_out, original)

            self.add_label("{},{},{}".format(number_frame, position[0], position[1]))

        exam.release()
        cv2.destroyAllWindows

    def run(self):
        files = os.listdir(self.dataset_path)
        for file in files:
            self.title = file.replace('.mp4', '')
            self.add_label('frame,x,y')

            self.dataset_out_exam = r'{}\{}'.format(self.dataset_out, self.title)
            os.mkdir(self.dataset_out_exam)

            exam = cv2.VideoCapture('{}/{}'.format(self.dataset_path, file))
            self.pupil_process(exam)


main = Main()
if __name__ == '__main__':
    main.run()
