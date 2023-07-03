import moviepy.editor as mp
import os
from pyannote.audio import Pipeline
from deepface import DeepFace
import cv2
import numpy as np


def clean_str(str):
    str = str.replace(':','')
    str = str.replace('.','')
    return int(str)

base_path = '/Aphasia/'
base_post_processed_path = 'post_processed/'
options = ['Control/', 'Aphasia/']

# The auth token is individual and is required to be asked to use in the Hugging face page
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="")

for option in options:
    parent_path = base_path + option
    for folder_path in os.scandir(parent_path):
        if folder_path.is_dir():
            paths = folder_path.path.split('/')
            folder_name = paths[len(paths) - 1] + '/'
            for file in os.listdir(folder_path.path):
                cnt_no_detectadas = 0
                if 'mp4' in file:
                    video_path = folder_path.path + '/' + file

                    post_processed_path = base_path + base_post_processed_path + option + folder_name

                    if not os.path.exists(post_processed_path):
                        os.mkdir(post_processed_path)

                    post_processed_path += file[:-4] + '/'

                    if not os.path.exists(post_processed_path):
                        os.mkdir(post_processed_path)

                        diarization_path = post_processed_path + 'diarization.txt'
                        emotion_total_path = post_processed_path + "emotions_total.txt"
                        emotion_dominant_path = post_processed_path + "emotions_dominant.txt"
                        no_detectadas_path = post_processed_path + "ratio_no_detectadas.txt"

                        audio_path = post_processed_path + file[:-3] + 'wav'

                        my_clip = mp.VideoFileClip(video_path)
                        my_clip.audio.write_audiofile(audio_path)
                        os.system('whisper ' + audio_path + ' --language English --verbose False --output_dir ' + post_processed_path)


                        diarization = str(pipeline(audio_path, num_speakers=2))

                        with open(diarization_path, "w") as text_file:
                            text_file.write(diarization)

                        diarization = open(diarization_path, "r")
                        diarization_lines = diarization.readlines()
                        diarization.close()

                        cap = cv2.VideoCapture(video_path)

                        if (cap.isOpened()== False):
                            print("Error opening video stream or file")

                        inicios_diarization = []
                        finales_diarization = []
                        pacientes_diarization = []

                        for line in diarization_lines:
                            inicio = clean_str(line[2:14])
                            final = clean_str(line[20:32])

                            inicios_diarization.append(inicio)
                            finales_diarization.append(final)
                            pacientes_diarization.append(line.find('SPEAKER_00') != -1)

                        inicios_diarization_filtrado = []
                        finales_diarization_filtrado = []
                        pacientes_diarization_filtrado = []

                        i = 0
                        cambio = True
                        while i < len(inicios_diarization) - 1:

                            if cambio:
                                inicios_diarization_filtrado.append(inicios_diarization[i])
                                pacientes_diarization_filtrado.append(pacientes_diarization[i])
                                cambio = False

                            if pacientes_diarization[i] != pacientes_diarization[i+1]:

                                finales_diarization_filtrado.append(finales_diarization[i])
                                cambio = True

                                if i == len(inicios_diarization) - 2:
                                    inicios_diarization_filtrado.append(inicios_diarization[i+1])
                                    finales_diarization_filtrado.append(finales_diarization[i+1])
                                    pacientes_diarization_filtrado.append(pacientes_diarization[i+1])
                            else:
                                if i == len(inicios_diarization) - 2:
                                    finales_diarization_filtrado.append(finales_diarization[i+1])
                            i += 1


                        tmp_0 = 0
                        tmp_1 = 0

                        for i in range(len(inicios_diarization_filtrado)):
                            if i % 2 == 0:
                                tmp_0 += finales_diarization_filtrado[i] - inicios_diarization_filtrado[i]
                            else:
                                tmp_1 += finales_diarization_filtrado[i] - inicios_diarization_filtrado[i]

                        # FALSE -> HABLA EL INTERVIEWER

                        i = 0
                        if tmp_1 > tmp_0:
                            i = 1

                        final_frames = []

                        seconds = 0.2
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        multiplier = int(round(fps * seconds))

                        frameID = int(round(inicios_diarization_filtrado[i] / 1000 * fps))
                        idx_frame = frameID
                        final_frame = int(round(inicios_diarization_filtrado[i+1] / 1000 * fps))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
                        ret, frame = cap.read()
                        final_frames.append(frame)

                        cnt = 0
                        datas = []
                        while(ret):
                            if idx_frame > final_frame:
                                if i+2 < len(inicios_diarization_filtrado):
                                    i += 2
                                    frameID = int(round(inicios_diarization_filtrado[i] / 1000 * fps))
                                    idx_frame = frameID
                                    if i + 1 < len(inicios_diarization_filtrado):
                                        final_frame = int(round(inicios_diarization_filtrado[i+1] / 1000 * fps))
                                    else:
                                        final_frame = int(round(finales_diarization_filtrado[i] / 1000 * fps))
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
                                else:
                                    ret = False
                            else:
                                ret, frame = cap.read()
                                if ret == True:
                                    if idx_frame % multiplier == 0:
                                        try:
                                            data = DeepFace.analyze(frame, actions=['emotion'])
                                            datas.append(data[0])
                                            tf.keras.backend.clear_session()
                                        except:
                                            cnt_no_detectadas += 1


                                    idx_frame += 1

                        f = open(emotion_total_path, "w")
                        for data in datas:
                            f.write(str(data))
                        f.close()

                        fd = open(emotion_dominant_path, "w")
                        for data in datas:
                            fd.write(data['dominant_emotion'] + '\t' + str(data['emotion'][data['dominant_emotion']]) + '\n')
                        fd.close()

                        fn = open(no_detectadas_path, "w")
                        fn.write(str(cnt_no_detectadas) + '\t' + str(cnt_no_detectadas + len(datas)) + '\n')
                        fn.close()