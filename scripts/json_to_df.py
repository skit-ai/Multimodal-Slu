"""
Usage:
    json-process --input_json=<input_json> --audio_dir=<audio_dir>  

Options:
  --input_json = <input_json> ..... input json to be used
  --audio_dir = <audio-dir> .... output audio dir ... this where downloaded files will be stored 
"""

import ast
import os
import json
from docopt import docopt

import uuid

import requests
from urllib.parse import urlparse
import pandas as pd 
from tqdm import tqdm 

from torchaudio.sox_effects import apply_effects_file


EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]

def audio_url_to_file(url,directory_name):
    path = urlparse(url).path
    file_name = str(path[1:])
    file_name = os.path.split(path)[-1]
    if not os.path.exists(directory_name + "/" + file_name):
        r = requests.get(url, allow_redirects=True)
        save_path = directory_name + "/" + file_name
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
    audio_dir = args["--audio_dir"]
    if os.path.exists(audio_dir):
        os.remove(audio_dir)
    os.mkdir(audio_dir)
    with open(input_json,'r') as f:
        rows = json.load(f)

    input_df = []
    num = 0
    for i,row in tqdm(enumerate(rows)):
        audio_url = row["data"]["audio_url"]
        print(audio_url)
        intent_label = row["intent"]
        alternatives = "</s></s>".join([x['transcript'] for x in row["data"]["alternatives"][0]])
        path = audio_url_to_file(audio_url,audio_dir)
        path = flac2wav(path)
        wav, _ = apply_effects_file(str(path), EFFECTS)
        base_path = "scripts/"
        input_df.append([base_path+path,intent_label,alternatives])
        '''
        except:
            num += 1
            print(i)
            pass  
        '''
    input_df = pd.DataFrame(input_df,columns=["audio_path","label","alternatives"])           
    input_df.to_csv(f"{audio_dir}.csv", index=False)
    print("total audios skipped :", num)

if __name__ == '__main__':
    main()
