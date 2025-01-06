import os
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ROOT = '//'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
DATA_PATH = 'N:\PROJECTS\python\STUDY\SHADOW\DATASET\DATA_01'
