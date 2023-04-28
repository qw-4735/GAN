#%%
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
#z_dim = 100
#%%
# Generator : 랜덤벡터 z를 입력으로 받아 가짜 이미를 출력하는 함수
# Discriminator : 이미지를 입력으로 받고, 그 이미지가 진짜일 확률을 0과 1 사이의 값으로 출력하는 함수.

#fake_data = G(z.to(device)).detach()  # [64, 1, 28, 28]

"""Generator"""
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, inputs):  # (batch_size x z_size) 크기의 랜덤 벡터를 받아 (batch_size x 1 x 28 x 28)크기의 이미지를 출력
        return self.main(inputs).view(-1,1,28,28)    

"""Discriminator"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__() 
        self.n_input = 28*28   # 판별자의 입력 크기
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),   
            nn.Linear(1024, 512),
            nn.LeakyReLU(0,2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # overfitting 방지, 차별자가 생성자보다 지나치게 빨리 학습되는 것을 막기 위함 
            nn.Linear(256, 1),  # 마지막에는 확률값을 나타내는 숫자 하나를 출력
            nn.Sigmoid()  # 출력값을 0과 1 사이로 만들기 위함
            )   
    # (batch_size x 1 x 28 x 28) 크기의 이미지를 받아 이미지가 진짜일 확률을 0~1 사이로 출력
    def forward(self, inputs):
        inputs =  inputs.view(-1, 28*28)
        return self.main(inputs)    
#%%

        
            
            
            
            
            
            
            
    