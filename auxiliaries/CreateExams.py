import pandas as pd
import os
import cv2


def create_exam(list_frames, path_writer, size_writer, fps):
    out = cv2.VideoWriter(path_writer, cv2.VideoWriter_fourcc(*'DIVX'), fps, size_writer)

    for i in range(len(list_frames)):
        out.write(img_array[i])

    out.release()


exam = ''

path_original_exams = '/Projects/Datasets/Exams'
# path_original_exams = '/media/marcos/Dados/Projects/Datasets/Exams'

path_exams = '/Projects/Results/PupilDeep/Frames'
# path_exams = '/media/marcos/Dados/Projects/Results/PupilDeep/Frames/not_use.'

path_labels = '/Projects/Results/PupilDeep/Labels'
# path_labels = '/media/marcos/Dados/Projects/Results/PupilDeep/Labels/not_use.'

path_out = '/Projects/Results/PupilDeep/Exams'
# path_out = '/media/marcos/Dados/Projects/Results/PupilDeep/Exams'

print(path_original_exams, '\n', path_exams, '\n', path_labels, '\n', path_out)

paths_frames = []
if exam == '':
    paths_frames = ['{}/{}'.format(path_exams, x) for x in os.listdir(path_exams) if '.' not in x]
else:
    paths_frames.append('{}/{}'.format(path_exams, exam))

print(paths_frames[0:5])
print(len(paths_frames))

img_array = []
size = None
for ex in paths_frames:
    name_exam = ex[ex.rfind('/') + 1 : len(ex)]    
    path_writer = '{}/{}.mp4'.format(path_out, name_exam)

    path_original_exam = '{}/{}.mp4'.format(path_original_exams, name_exam)    
    original_exam = cv2.VideoCapture(path_original_exam)
    fps = original_exam.get(cv2.CAP_PROP_FPS)

    label = '{}/{}_label.csv'.format(path_labels, name_exam)
    print(label)
    dataset = pd.read_csv(label)

    number_files = int(len(os.listdir(ex)) / 5)
    for i in range(number_files):       
        path_img = 'img_process_{}.png'.format(i)
        img = cv2.imread('{}/{}'.format(ex, path_img))
        img_array.append(img)

        if size is None:
            height, width, _ = img.shape
            size = (width, height)

    create_exam(img_array, path_writer, size, fps)
    print("finish exam: {}, writer in: {}".format(name_exam, path_writer))
    img_array.clear()
