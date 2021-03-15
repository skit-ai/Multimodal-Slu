import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim,cls_dimension=768, output_class_num, **kwargs):
        super(Model, self).__init__()
        hidden_dim = kwargs["hidden_dim"]
        self.connector = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        #Add cls dimension from text model 
        self.fc2 = nn.Linear(hidden_dim + cls_dimension, output_class_num)

    def forward(self, audio_features,text_features):
        features = F.relu(self.connector(features))
        features = self.fc1(features)
        pooled = features.mean(dim=1)
        audio_text_feature = torch.cat([pooled,text_features],dim=1)
        predicted = self.fc2(audio_text_feature)
        return predicted
