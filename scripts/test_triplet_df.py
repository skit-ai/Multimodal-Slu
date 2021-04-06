import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')


def convert_to_triplet_df(df):
    # ideal way is to create triplets on runtime, also keep unknown label class 
    # to check semantic similarity, but for now lets create triplet df.
    wavs, labels = df['audio_path'], df['label']
    wavs = wavs.tolist()
    labels = labels.tolist()
    labels_set = set(labels)
    label_to_wavs = {}
    for label in labels_set:
        wav = []
        for i in range(len(labels)):
            if labels[i] == label:
                wav.append(wavs[i])
        label_to_wavs[label] = wav

    random_state = np.random.RandomState(29)
    triplets = [
        [
            i, # anchor
            random_state.choice(label_to_wavs[labels[i]]),
            random_state.choice(label_to_wavs[np.random.choice(
                list(labels_set - set([labels[i]]))
            )
        ])
    ]
    for i in range(len(wavs))]
    return triplets

triplet_df = convert_to_triplet_df(df)
print(triplet_df)