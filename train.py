from dataProcessing import *
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
sentences, tags = ReadTrain()
training_data = [(sentences, tags)]






