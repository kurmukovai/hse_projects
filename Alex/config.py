
#config
import torch
from pathlib import Path
to_one_class = True 
num_epochs = 100
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
data_folder = Path("/home/alexey/Thesis/MICCAI_BraTS2020_TrainingData")
continue_train = False

training_ratio = 0.7 
validation_ratio = 0.2
test_ratio = 0.1
