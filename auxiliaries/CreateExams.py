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
path_exams = '/Projects/Results/PupilDeep/Frames'
path_labels = '/Projects/Results/PupilDeep/Labels'
path_out = '/Projects/Results/PupilDeep/Exams'.format(path_exams)

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
        path_final = 'final_{}.png'.format(i)

        im_final = cv2.imread('{}/{}'.format(ex, path_final))

        im_pres = im_final

        label = 'Frame=%d;Radius=%d;Center=(%d,%d);Eye=(%d)' % (dataset['frame'][i],
                                                                dataset['radius'][i],
                                                                dataset['center_x'][i],
                                                                dataset['center_y'][i],
                                                                0)

        cv2.putText(im_pres, label, (600, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (170, 170, 0))

        img_array.append(im_pres)

        if size is None:
            height, width, _ = im_pres.shape
            size = (width, height)

    create_exam(img_array, path_writer, size, fps)
    print("finish exam: {}, writer in: {}".format(name_exam, path_writer))
    img_array.clear()
