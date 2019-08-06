import torch.nn as nn
import torch
import torch.nn.functional as F


def build_mlp(dataloaders, attrib_dict, device):
    torch.manual_seed(1)
    attrib = attrib_dict['train']
    model = MLPClassifier(attrib.feature_length, 512,
                          attrib.num_classes, dataloaders['train'], device)
    return model


class MLPClassifier(nn.Sequential):
    '''Defines a multilayer perceptron with one hidden layer for classification.
    Attributes:
        input_size (int): Number of units in the input layer, a.k.a. number of features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of units in the output layer, a.k.a. number of classes.
        dataloader (torch.utils.data.DataLoader): Holds the data that the model will train on.
        device (torch.device): an object representing whether the model runs on cpu or cuda.
    '''

    def __init__(self, input_size, hidden_size, output_size, dataloader, device):
        super(MLPClassifier, self).__init__(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_size, output_size)
        )

        self.to(device)
        self.hidden_size = hidden_size
        self.dataloader = dataloader
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
