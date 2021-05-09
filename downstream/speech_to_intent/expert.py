import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from downstream.speech_to_intent.model import Model
from downstream.speech_to_intent.dataset import SpeechCommandsDataset
import pandas as pd
from collections import Counter
import IPython
from sklearn.model_selection import train_test_split
import yaml
import torch.nn.functional as nnf

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
        self.intent_alias_yaml = downstream_expert['datarc'].get('intent_alias')
        self.alias_config = None 
        self.min_class_count = downstream_expert['datarc']['min_class_count']
        train_df = pd.read_csv(self.datarc['train_file_path'])[["audio_path","label"]]
        print(train_df.head())
        test_df =  pd.read_csv(self.datarc['test_file_path'])[["audio_path","label"]]
        dev_df = pd.read_csv(self.datarc['dev_file_path'])[["audio_path","label"]]
        if self.intent_alias_yaml is not None:
            with open(self.intent_alias_yaml) as file:
                self.alias_config = yaml.load(file, Loader=yaml.FullLoader)
            train_df = self.intent_alias(train_df)
            test_df = self.intent_alias(test_df)
            dev_df = self.intent_alias(dev_df)  

        classes = train_df.label.unique().tolist()
        #rint(classes)
        import downstream.speech_to_intent.dataset as ds
        setattr(ds, 'CLASSES', classes)
        # add logic here to remove classes
        #train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
        #train_df = test_df = df
        train_df.to_csv(f"data/train_{self.min_class_count}.csv")
        test_df.to_csv(f"data/test_{self.min_class_count}.csv")
        train_list = train_df.values.tolist()
        test_list = test_df.values.tolist()
        dev_list = dev_df.values.tolist()
        
        self.train_dataset = SpeechCommandsDataset(train_list, classes, **self.datarc)
        self.dev_dataset = SpeechCommandsDataset(dev_list, classes, **self.datarc)
        self.test_dataset = SpeechCommandsDataset(test_list, classes, **self.datarc)
        
        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])
        
        self.model = Model(input_dim=self.modelrc['input_dim'], agg_module=self.modelrc['agg_module'],output_dim=len(classes), config=self.modelrc)
        self.objective = nn.CrossEntropyLoss()


    def intent_alias(self,df):
        df['label'] =  df['label'].apply(lambda x:self.map_intent(x,self.alias_config))
        df = df.dropna() 
        df = df[df.groupby('label')['label'].transform('count').ge(self.min_class_count)]
        return df

    def map_intent(self,intent,intent_config):
        if intent_config is None:
            return intent
        if intent_config.get("intent_alias"):    
            for intent_key in intent_config["intent_alias"].keys():
                for candidate_intent in intent_config["intent_alias"][intent_key]:
                    if (candidate_intent == intent) and (intent_key in intent_config["pick_intents"]):
                        return intent_key
        if intent in intent_config["pick_intents"]:    #return intent when intent is in pick intents 
            return intent

        return None      # return None when intent not in pick intents     


    def get_dataset(self):
        self.base_path = self.datarc['file_path']
        train_df = pd.read_csv(os.path.join(self.base_path, "nlu_iob", "iob.train"), sep='\t', header=None)
        valid_df = pd.read_csv(os.path.join(self.base_path, "nlu_iob", "iob.dev"), sep='\t', header=None)
        test_df = pd.read_csv(os.path.join(self.base_path, "nlu_iob", "iob.test"), sep='\t', header=None)

        train_dict = {"id": {}, "intent": {}}
        valid_dict = {"id": {}, "intent": {}}
        test_dict = {"id": {}, "intent": {}}

        
        for dc, df, type_name in [(train_dict, train_df, 'train'), (valid_dict, valid_df, 'dev'), (test_dict, test_df, 'test')]:
            n = 0
            for i in range(len(df)):
                if(os.path.exists( os.path.join(self.base_path, type_name, df[0][i].split()[0]+'.wav')) & (df[0][i].split()[0] not in dc['id'].values())):
                    dc['id'][n] = df[0][i].split()[0]
                    dc['intent'][n] = df[1][i].split()[-1]
                    n += 1
                # else:
                #     print(os.path.join(self.base_path, type_name, df[0][i].split()[0]+'.wav'))
            print(type_name, ': ', n+1)    
            
        #IPython.embed()
        train_df = pd.DataFrame(data=train_dict)
        valid_df = pd.DataFrame(data=valid_dict)
        test_df = pd.DataFrame(data=test_dict)


        Sy_intent = {"intent": {}}
        values_per_slot = []
        
        # IPython.embed()
        
        for slot in ["intent"]:
            slot_values = Counter(train_df[slot]) + Counter(valid_df[slot]) + Counter(test_df[slot])
            for index, value in enumerate(slot_values):
                Sy_intent[slot][value] = index
            values_per_slot.append(len(slot_values))
        self.values_per_slot = values_per_slot
        self.Sy_intent = Sy_intent
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        # IPython.embed()

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

    # Interface
    def forward(self, features, labels, audio_paths,
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
        features_pad = pad_sequence(features, batch_first=True)
        
        attention_mask = [torch.ones((feature.shape[0])) for feature in features] 

        attention_mask_pad = pad_sequence(attention_mask,batch_first=True)

        attention_mask_pad = (1.0 - attention_mask_pad) * -100000.0

        features_pad = self.connector(features_pad)
        intent_logits = self.model(features_pad, attention_mask_pad.cuda())
        intent_loss = 0
        start_index = 0
        predicted_intent = []
        labels = torch.LongTensor(labels).to(features_pad.device)
        
        #labels = torch.stack(labels).to(features_pad.device)
        '''for slot in range(len(self.values_per_slot)):
            end_index = start_index + self.values_per_slot[slot]'''
        #subset = intent_logits[:, start_index:end_index]
        intent_loss += self.objective(intent_logits, labels)
        predicted_intent.append(intent_logits.max(1)[1])
        predicted_intent = torch.stack(predicted_intent, dim=1)
        #IPython.embed()
        records['acc'] += (predicted_intent == labels).prod(1).view(-1).cpu().float().tolist()

        if not self.training:
            # some evaluation-only processing, eg. decoding
            #IPython.embed()
            pass
        prob = nnf.softmax(intent_logits, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)    
        #print(predicted_intent.cpu().float().tolist(),labels.cpu().float().tolist())
        return intent_loss,predicted_intent.cpu().float().tolist(),labels.cpu().float().tolist(),audio_paths,top_p.cpu().float().tolist()

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
