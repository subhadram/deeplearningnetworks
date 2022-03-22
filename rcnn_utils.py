import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    file1 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment2/datasets/train_data_x.pkl', 'rb')
    file2 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment2/datasets/train_data_y.pkl', 'rb')
    file3 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment2/datasets/test_data_x.pkl', 'rb')
    file4 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment2/datasets/test_data_y.pkl', 'rb')
    trnx = pkl.load(file1,encoding='latin1')
    trny = pkl.load(file2,encoding='latin1')
    tstx = pkl.load(file3,encoding='latin1')
    tsty = pkl.load(file4,encoding='latin1')
    return(trnx,trny,tstx,tsty)

# CNN model
#class Net(nn.Module):
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6,[5,5],stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6,[1,1],stride=1)
        self.conv3 = nn.Conv2d(6, 6,[1,1],stride=1)
        self.conv4 = nn.Conv2d(6, 12, [5,5],stride=1)
        self.fc1 = nn.Linear(12 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 64)
        self.dropout = nn.Dropout(0.2)
        self.downsample = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5,stride=2,bias=False))
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        residual = x
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        #print(x.size())
        residual = self.downsample(residual)
        #print(residual.size())
        x = x + residual
        x = F.relu(self.conv3(x))
        #print(x.size())
        #residual = self.downsample(residual)
        #print(residual.size())
        x = x + residual
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        #print(x.size())
        x = x.view(-1, 12 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

"""    
class Net(nn.Module):
    def __init__(self):
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, [5,5],stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, [5,5],stride=1)
        self.fc1 = nn.Linear(12 * 13 *13, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 12 * 13 *13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
model = Net().to(device)
model.train()
print(Net)





