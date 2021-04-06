import os
import math
import torch
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from downstream.speech_utterance_embedding.model import Model
from downstream.speech_utterance_embedding.dataset import SpeechCommandsDataset
import pandas as pd
from collections import Counter
import IPython
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension
            
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. benchmark/downstream/example/config.yaml

            **kwargs: dict
                The arguments specified by the argparser in run_benchmark.py
                in case you need it.
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        df = pd.read_csv(self.datarc['file_path'])
        triplet_df =  self.convert_to_triplet_df(df)
        # add logic here to remove classes
        train_df, test_df = train_test_split(triplet_df, test_size=0.15, random_state=42)
        train_list = train_df.values.tolist()
        valid_list = test_df.values.tolist()

        self.train_dataset = SpeechCommandsDataset(train_list, **self.datarc)
        self.dev_dataset = SpeechCommandsDataset(valid_list, **self.datarc)
        self.test_dataset = SpeechCommandsDataset(valid_list, **self.datarc)

        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])

        self.model = Model(input_dim=self.modelrc['input_dim'], agg_module=self.modelrc['agg_module'], config=self.modelrc)
        self.objective = nn.CrossEntropyLoss()
        self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)


    def convert_to_triplet_df(self, df):
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
                wavs[i], # anchor
                random_state.choice(label_to_wavs[labels[i]]), # positive
                random_state.choice(label_to_wavs[np.random.choice(
                    list(labels_set - set([labels[i]]))
                ) # negative
            ])
        ]
        for i in range(len(wavs))]
        return pd.DataFrame(triplets)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=True, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    """
    Datalaoder Specs:
        Each dataloader should output a list in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with:
            1. dim() == 1
            2. sample_rate == 16000
            3. directly loaded by torchaudio without any preprocessing
    """

    # Interface
    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    # Interface
    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    # Interface
    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    
    #Function to handle padding of features 
    def pad_feature(self,features):
        features_pad = pad_sequence(features, batch_first=True)
        
        attention_mask = [torch.ones((feature.shape[0])) for feature in features] 

        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)

        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)

        return features_pad, attention_mask_pad


    def forward(self, features, labels,
                records=None, logger=None, prefix=None, global_step=0, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by dumping contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records

                Note1. benchmark/runner.py will call self.log_records
                    1. every log_step during training
                    2. once after evalute the whole dev/test dataloader

                Note2. log_step is defined in your downstream config

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging convenience
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        anchor_features,pos_features,neg_features = features
        anchor_features_pad,anchor_attention_mask_pad,pos_features_pad,pos_attention_mask_pad,neg_features_pad,neg_attention_mask_pad = self.pad_feature(anchor_features),self.pad_feature(pos_features),self.pad_feature(neg_features)
        rep_anchor, rep_pos, rep_neg = self.model(anchor_features_pad,anchor_attention_mask_pad.cuda()),self.model(pos_features_pad,pos_attention_mask_pad.cuda()),self.model(neg_features_pad,neg_attention_mask_pad.cuda())

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        #records['acc'] += (predicted_intent == labels).prod(1).view(-1).cpu().float().tolist()
        return losses.mean(),rep_anchor, rep_pos, rep_neg 

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'{prefix}{key}',
                average,
                global_step=global_step
            )

        if not self.training:
            # some evaluation-only processing, eg. decoding
            pass
