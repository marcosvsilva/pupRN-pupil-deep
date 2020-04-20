import math


class Information:
    def __init__(self):
        self._frames_flash_information = []
        self._stimulus = []

    def _calc_frame_stimulus(self, time_stimulus, fps_movie):
        stimulus = time_stimulus
        minutes = int(stimulus[0:stimulus.find(':')])
        seconds = int(stimulus[stimulus.find(':') + 1:stimulus.rfind(':')])
        milliseconds = int(stimulus[stimulus.rfind(':') + 1:len(stimulus)])
        frame = math.ceil((minutes * (fps_movie * 60)) + (seconds * fps_movie) + ((milliseconds * fps_movie) / 100))
        return frame

    def _calc_ranges_stimulus(self, stimulus):
        list_all_stimulus = []
        for st in stimulus:
            init_frame, end_frame = st['init_stimulus'], st['end_stimulus']
            list_range = [i for i in range(int(init_frame), int(end_frame), 1)]
            list_all_stimulus.append({'stimulus': st['stimulus'], 'frames': list_range})

        self._frames_flash_information = [y for x in [x['frames'] for x in list_all_stimulus] for y in x]
        return list_all_stimulus

    def get_information_params(self, number_frame):
        flash_information, color_information = 0, ''
        if number_frame in self._frames_flash_information:
            for st in self._stimulus:
                if number_frame in st['frames']:
                    flash_information = 1
                    color_information = st['stimulus']
                    break

        return flash_information, color_information

    def get_information_exam(self, path_read_information, fps_movie):
        patient, param, stimulus = '', '', []
        with open(path_read_information, 'r') as information:
            activate = False
            for line in information.readlines():
                if 'PAC' in line:
                    patient = line[line.rfind(':') + 2:len(line) - 1]
                if 'PRM' in line:
                    param = line[line.rfind(':') + 2:len(line) - 1]
                    param = param.replace(',', '-')
                    param = param.replace(' ', '')
                if 'ATV' in line:
                    color_stimulus = line[line.rfind(':') + 2:len(line) - 1]
                    init_stimulus = self._calc_frame_stimulus(line[0:line.rfind('-') - 1], fps_movie)
                    activate = not activate
                if 'DTV' in line:
                    if activate:
                        activate = not activate
                        end_stimulus = self._calc_frame_stimulus(line[0:line.rfind('-') - 1], fps_movie)
                        stimulus.append({'stimulus': color_stimulus,
                                         'init_stimulus': init_stimulus,
                                         'end_stimulus': end_stimulus})

        self._stimulus = self._calc_ranges_stimulus(stimulus)
        return patient, param
