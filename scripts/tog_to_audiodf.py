"""
Usage:
    tog-job-process --input_sqlite=<input_sqlite> --job_id=<job_id>  

Options:
  --input_sqlite = <input_sqlite> ..... input sqlite to be used
  --job_id = <job_id> .... job id directory name will be used 
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

def audio_url_to_file(url,directory_name):
    path = urlparse(url).path
    file_name = str(path[1:])
    if not os.path.exists(directory_name + "/" + file_name):
        r = requests.get(url, allow_redirects=True)
        save_path = directory_name + "/" + file_name
        open(save_path, 'wb').write(r.content)
        return save_path
    return None     

def main():
    args = docopt(__doc__)
    input_sqlite = args["--input_sqlite"]
    job_id = args["--job_id"]
    if not os.path.exists(job_id):
        os.mkdir(job_id)
    conn = sqlite3.connect(input_sqlite)
    cur = conn.cursor()
    cur.execute("SELECT * FROM data")
    rows = cur.fetchall()
    input_df = []
    for i,row in tqdm(enumerate(rows)):
        try:
            audio_url = json.loads(row[1])["audio_url"]
            intent_label = json.loads(row[2])[0]["type"]
            audio_url_to_file(audio_url,job_id)
            input_df.append([audio_url,intent_label])
        except:
            print(i)
            pass  
    input_df = pd.DataFrame(input_df,columns=["audio_url","label"])           
    input_df.to_csv(f"{jod_id}.csv")

if __name__ == '__main__':
    main()
