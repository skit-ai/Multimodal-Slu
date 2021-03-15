import os


mypath = "/home/ml/karthik/s3prl/corpora/speech_commands_v0.01/other"

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

import glob

for p in glob.glob(mypath+"/*.flac"):
    new_p = flac2wav(p)
    print(new_p)