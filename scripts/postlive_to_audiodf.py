"""
Usage:
    post-live-process --input_json=<input_json> --dir_name=<dir_name>  

Options:
  --input_json = input json
  --dir_name = dir name
"""

import ast
import os
import json
from docopt import docopt

import uuid

import sqlite3
import requests
from urllib.parse import urlparse
import pandas as pd 
from tqdm import tqdm 

from torchaudio.sox_effects import apply_effects_file


EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]

def audio_url_to_file(url,directory_name):
    path = urlparse(url).path
    file_name = str(path[1:])
    if not os.path.exists(directory_name + "/" + file_name):
        r = requests.get(url, allow_redirects=True)
        save_path = directory_name + "/" + file_name
        file_paths = file_name.split('/')
        if len(file_paths) > 1:
            sub_dir = '/'.join(file_paths[:-1])
            if not os.path.exists(directory_name + "/" + sub_dir):
                os.makedirs(directory_name + "/" + sub_dir)
        open(save_path, 'wb').write(r.content)
        return save_path
    else:
        save_path = directory_name + "/" + file_name
        return save_path  

def flac2wav(flac_path):
    # check if flac:
    file_name, ext = os.path.splitext(flac_path)
    if ext == '.flac':
        command = "ffmpeg -i " + flac_path + " " + file_name + ".wav" 
        os.system(command)
        os.system("rm -rf " + flac_path)
        new_path = file_name + ".wav"
        return new_path
    else:
        return flac_path

def main():
    args = docopt(__doc__)
    input_json = args["--input_json"]
    dir_name = args["--dir_name"]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    input_df = []
    num = 0

    with open(input_json) as f:
        data = json.load(f)
    
    for d in data:
        audio_url = 'https://hermes-cca.s3.ap-south-1.amazonaws.com/' + d['audio_url_key']
        intent = d['intent']
        alternatives = d['alternatives']
        path = audio_url_to_file(audio_url,dir_name)
        path = flac2wav(path)
        print(path)
        wav, _ = apply_effects_file(str(path), EFFECTS)
        base_path = "scripts/"
        input_df.append([base_path+path,intent])
     
    input_df = pd.DataFrame(input_df,columns=["audio_path","label"])           
    input_df.to_csv(f"{dir_name}.csv", index=False)

if __name__ == '__main__':
    main()
