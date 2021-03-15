import pandas as pd
import os
import shutil


data = pd.read_csv('/home/ml/karthik/s3prl/tog_job/437.csv')

root = '/home/ml/karthik/s3prl/tog_job/437_audios'
dest_cancel = '/home/ml/karthik/s3prl/corpora/speech_commands_v0.01/cancel'
dest_confirm = '/home/ml/karthik/s3prl/corpora/speech_commands_v0.01/confirm'
dest_other = '/home/ml/karthik/s3prl/corpora/speech_commands_v0.01/other'
'''
def flac2wav(flac_path):
    # check if flac:
    file_name, ext = os.path.splitext(flac_path)
    if ext == '.flac':
        command = "ffmpeg -i " + flac_path + " " + file_name + ".wav" 
        os.system(command)
        new_path = file_name + ".wav"
        return new_path
    else:
        return flac_path
'''
for index, rows in data.iterrows():
    file = rows['audio_url']
    #file = flac2wav(root+file[35:])
    label = rows['label']
    if rows['label'] == 'confirm':
        shutil.copy(root+file[35:], dest_confirm)
    elif rows['label'] == 'cancel':
        shutil.copy(root+file[35:], dest_cancel)
    else:
        shutil.copy(root+file[35:], dest_other)
