import pupil_deep
import cv2
import os
import numpy as np
import csv


def pupil_process(exam):
    eye_tracker = pupil_deep.DeepEye()
    number_frame = 0

    while True:
        _, frame = exam.read()

        if frame is None and number_frame > 30:
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

        position = eye_tracker.run(dilate)

        lin, col = gray.shape
        if 0 < position[0] < lin and 0 < position[1] < col:
            cv2.circle(original, (int(position[0]), int(position[1])), 10, (255, 255, 0), 2)

        text = "frame={}, x={}, y={}".format(number_frame, position[0], position[1])
        cv2.putText(original, text, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        number_frame += 1
        path_out = "{}\\{}.png".format(dataset_out, number_frame)

        cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
        cv2.imshow('Analysis', original)
        cv2.waitKey(1)
        cv2.imwrite(path_out, original)

        with open(csv_dataset, 'a') as csvfile:
            csvfile.write("{},{},{}\n".format(number_frame, position[0], position[1]))
            csvfile.close()

    exam.release()
    cv2.destroyAllWindows


directory_path = os.getcwd().replace('\\pupil_deep', '')
dataset = directory_path + "\\dataset"
dataset_out = directory_path + "\\dataset_out"
csv_dataset = directory_path + "\\dataset\\dataset.csv"
exams = os.listdir(dataset)

with open(csv_dataset, 'a', newline='') as csvfile:
    csvfile.write("frame,x,y\n")
    csvfile.close()

for exam in exams:
    movie = cv2.VideoCapture("{}/{}".format(dataset, exam))
    pupil_process(movie)
