import os
from multiprocessing import Process
from execution import Execution


class Main:
    def __init__(self):
        # Directoris
        self._projects_path = '/Projects' 

        self._path_dataset = '{}/Datasets/Exams'.format(self._projects_path)
        self._path_information = '{}/Datasets/Exams/Information_Exams'.format(self._projects_path)
        self._path_out = '{}/Results/PupilDeep/Frames'.format(self._projects_path)
        self._path_label = '{}/Results/PupilDeep/Labels'.format(self._projects_path)

        # Exams to execute
        self._list_available = ['14080114_08_2019_07_56_28.mp4', '24070624_07_2019_09_25_42.mp4',
                                '030703_07_2019_08_09_08.mp4', '030703_07_2019_08_10_26.mp4',
                                '030703_07_2019_08_16_37.mp4', '0307403_07_2019_08_28_51.mp4',
                                '0307403_07_2019_08_30_18.mp4', '0307603_07_2019_08_42_26.mp4',
                                '07080107_08_2019_08_03_45.mp4', '07080107_08_2019_08_04_39.mp4',
                                '07080107_08_2019_08_05_48.mp4', '07080207_08_2019_08_11_53.mp4',
                                '07080207_08_2019_08_21_33.mp4', '07080407_08_2019_09_31_04.mp4',
                                '07080407_08_2019_09_33_39.mp4', '10070110_07_2019_08_44_16.mp4',
                                '10070210_07_2019_08_49_39.mp4', '10070210_07_2019_08_51_32.mp4',
                                '10070310_07_2019_08_57_56.mp4', '10070310_07_2019_09_00_02.mp4',
                                '110711_07_2019_08_03_05.mp4', '110711_07_2019_08_08_16.mp4',
                                '25080125_08_2019_08_32_38.mp4', '25080225_08_2019_08_37_59.mp4',
                                '25080225_08_2019_08_40_12.mp4', '25080325_08_2019_08_46_42.mp4',
                                '25080325_08_2019_08_48_58.mp4', '25080425_08_2019_08_53_48.mp4',
                                '25080425_08_2019_08_55_59.mp4', '25080425_08_2019_09_05_40.mp4',
                                '25080425_08_2019_09_08_25.mp4', '2906429_06_2019_13_34_56.mp4',
                                '31070231_07_2019_08_11_10.mp4', '31070231_07_2019_08_12_46.mp4',
                                'benchmark.mp4', 'new_benchmark.mp4', '14080314_08_2019_08_10_34.mp4',
                                '14080314_08_2019_08_12_38.mp4', '17070118_07_2019_08_02_25.mp4',
                                '17070218_07_2019_08_10_41.mp4', '17070218_07_2019_08_13_28.mp4',
                                '170717_07_2019_09_06_02.mp4', '170717_07_2019_09_08_12.mp4',
                                '21080321_08_2019_08_08_12.mp4', '24070124_07_2019_08_32_14.mp4',
                                '24070324_07_2019_08_58_46.mp4', '24070324_07_2019_09_00_37.mp4',
                                '24070424_07_2019_09_07_29.mp4', '24070424_07_2019_09_09_38.mp4',
                                '24070524_07_2019_09_16_00.mp4', '24070524_07_2019_09_18_30.mp4',
                                '24070624_07_2019_09_23_44.mp4', '14080214_08_2019_08_02_44.mp4',
                                '25080125_08_2019_08_28_50.mp4', '290629_06_2019_13_18_45.mp4',
                                '31070131_07_2019_08_06_31.mp4', '030703_07_2019_08_15_23.mp4',
                                '0307303_07_2019_08_21_00.mp4', '0307303_07_2019_08_22_21.mp4',
                                '0307603_07_2019_08_41_13.mp4', '07080107_08_2019_08_02_38.mp4',
                                '07080207_08_2019_08_09_49.mp4', '07080207_08_2019_08_19_12.mp4',
                                '10070110_07_2019_08_42_05.mp4', '110711_07_2019_08_07_56.mp4',
                                '14080114_08_2019_07_54_23.mp4', '14080114_08_2019_07_55_38.mp4',
                                '14080214_08_2019_08_05_17.mp4', '17070218_07_2019_08_13_07.mp4',
                                '21080121_08_2019_07_50_57.mp4', '21080121_08_2019_07_52_51.mp4',
                                '21080221_08_2019_07_58_59.mp4', '21080321_08_2019_08_06_09.mp4',
                                '24070124_07_2019_08_35_30.mp4', '24070224_07_2019_08_43_28.mp4',
                                '24070224_07_2019_08_46_51.mp4', '290629_06_2019_13_17_55.mp4',
                                '2906429_06_2019_13_35_53.mp4']

    def _make_path(self, path):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    def run(self):
        number_of_process = int(os.cpu_count()/2)

        exams_exists = ['{}.mp4'.format(x) for x in os.listdir(self._path_out)]
        if len(self._list_available) > 0:
            list_exams = [x for x in self._list_available if ('.mp4' in x) and (x not in exams_exists)]
        else:
            list_exams = [x for x in os.listdir(self._path_dataset) if ('.mp4' in x) and (x not in exams_exists)]

        process = []
        while len(list_exams) > 0:
            end_list = number_of_process if len(list_exams) > number_of_process else len(list_exams)
            list_process = list_exams[0:end_list]
            list_exams = list_exams[end_list:len(list_exams)]

            for exam in list_process:
                title = exam.replace('.mp4', '')

                paths = {'path_exam': '{}/{}'.format(self._path_dataset, exam),
                         'path_information':  '{}/{}.log'.format(self._path_information, title),
                         'path_out': '{}/{}'.format(self._path_out, title),
                         'path_label': '{}/{}_label.csv'.format(self._path_label, title)}

                self._make_path(paths['path_out'])

                execution = Execution()
                thread = Process(target=execution.pupil_process, args=(paths, ))
                process.append(thread)
                thread.start()

            for thread in process:
                thread.join()


main = Main()
if __name__ == '__main__':
    main.run()
