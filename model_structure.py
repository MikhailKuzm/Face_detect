from torch import nn
import torch

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels = 16, kernel_size = 3, stride =1,
                               padding= 1)
        self.conv2 = nn.Conv2d(16, out_channels = 32, kernel_size = 3, stride =1,
                               padding= 1)
        self.conv3 = nn.Conv2d(32, out_channels = 64, kernel_size = 3, stride =1,
                               padding= 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = (2, 2))
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(p=0.2)

        ### Первая голова (регрессия возраста)
        self.head_age = torch.nn.Sequential(
                        nn.Linear(2304, 128),
                        nn.BatchNorm1d(128),
                        nn.Dropout(p=0.2),
                        nn.ReLU(),
                        nn.Linear(128, 16),
                        nn.BatchNorm1d(16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                           )
        
        ### Вторая голова (мультиклассификация этнической принадлежности)
        self.head_eth = torch.nn.Sequential(
                        nn.Linear(2304, 256),
                        nn.BatchNorm1d(256),
                        nn.Dropout(p=0.2),
                        nn.ReLU(),
                        nn.Linear(256, 64),
                        nn.BatchNorm1d(64),
                        nn.Dropout(p=0.2),
                        nn.ReLU(),
                        nn.Linear(64, 16),
                        nn.BatchNorm1d(16),
                        nn.Linear(16, 5),
                        nn.ReLU()
                           )

        ### Третья голова (бинарная классификация пола)
        self.head_gender = torch.nn.Sequential(
                        nn.Linear(2304, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                           )

#Функция прямого прохода нейроннной сети 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.pool(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.pool(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.pool(x)
        x = self.drop(x)

        x = torch.flatten(x, 1)
        x_gen = self.head_gender(x)
        x_age = self.head_age(x)
        x_etn = self.head_eth(x) 
       
        return x_gen, x_age, x_etn


model = net()
