#%%
import os
os.getcwd()
os.chdir('D:\VAE\GAN')

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from utils.model import Generator, Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser(description='parameters')
    
    parser.add_argument('--num_epochs', type = int, default=30, help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--z_dim', default=128, type=int, help='dimension of the representation z')
    parser.add_argument('--k', default=1, type=float, help='the number of discriminator training')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
args = vars(get_args(debug=True))

#%%
transform = transforms.Compose([transforms.ToTensor(),   # 이미지 데이터를 Tensor형식으로 바꾼다.
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])  # 이미지 데이터 정규화
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=False, transform=transform)

train_dataloader =  DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True) # 데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.
test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)
images, image_labels =next(iter(train_dataloader))  # images.shape : [batch_size, 1, 28, 28] = [batch_size, channel, height, width]

# 이미지 시각화
import numpy as np
import matplotlib.pyplot as plt

torch_image = torch.squeeze(images[0])
torch_image.shape  # torch.Size([28, 28])

image = torch_image.numpy()
image.shape  # (28, 28)

label = image_labels[0].numpy()
label.shape
label

plt.title(label)
plt.imshow(image,'gray')
plt.show()


# import numpy as np
# from matplotlib import pyplot as plt

# def imshow(img):
#     img = (img+1)/2    
#     img = img.squeeze()
#     np_img = img.numpy()
#     plt.imshow(np_img, cmap='gray')
#     plt.show()

# def imshow_grid(img):
#     img = make_grid(img.cpu().detach())
#     img = (img+1)/2
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()

# example_mini_batch_img, example_mini_batch_label  = next(iter(train_dataloader))
# imshow_grid(example_mini_batch_img[0:16,:,:])

#%%
"""optimizer & loss function"""
G = Generator(args['z_dim']).to(device)
D = Discriminator().to(device) 

criterion = nn.BCELoss() # Binary Cross Entorpy loss

G_optimizer = Adam(G.parameters(), lr=args['lr'])
D_optimizer = Adam(D.parameters(), lr=args['lr'])   

#%%
"""판별자 학습을 위한 함수"""
def train_discriminator(optimizer, real_data, fake_data):
    batch_size = real_data.size(0)
    
    real_label = torch.ones(batch_size, 1).to(device)
    fake_label = torch.zeros(batch_size, 1).to(device)
    
    output_real = D(real_data)
    D_loss_real = criterion(output_real, real_label)
    
    output_fake = D(fake_data)
    D_loss_fake = criterion(output_fake, fake_label)
    
    D_loss = D_loss_real + D_loss_fake
    
    optimizer.zero_grad()  
    D_loss.backward()  
    optimizer.step() 
    
    return D_loss

 
   
def train_generator(optimizer, fake_data):
    batch_size = fake_data.size(0)
   
    real_label = torch.ones(batch_size, 1).to(device)  # 생성자 네트워크에서는 가짜 데이터만 사용하고 있는데, 생성자 입장에서는 가짜데이터가 실제로 진짜라는 것에 주의
    
    #z =  torch.randn((64,128), requires_grad=True)
    #fake_data = G(z.to(device)).detach()  # [64, 1, 28, 28]
    output_fake = D(fake_data)
    #main(fake_data).view(-1,1,28,28) 
    G_loss = criterion(output_fake, real_label)
    
    optimizer.zero_grad()  
    G_loss.backward()  
    optimizer.step()
    
    return G_loss


def save_generator_image(image,path):
    save_image(image, path)


#%%     
for epoch in range(args['num_epochs']):
    losses_g = []  # 매 에포크마다 발생하는 생성자 오차를 저장하기 위한 리스트
    losses_d = []
    images = []  # '생성자에 의해 생성되는 이미지'를 저장하기 위한 리스트    
    
    loss_g = 0.0
    loss_d = 0.0
    
    for idx, data in tqdm(enumerate(train_dataloader)):
        image, image_label = data  # 학습을 위한 이미지 데이터를 가져옴
        image = image.to(device)
        batch_size = len(image)  # 64
        
        for step in range(args['k']):
            z = torch.randn((batch_size, args['z_dim']), requires_grad=True) # [64, 128] # 생성자에 입력이 될 랜덤 백터 z 생성
            #z =  torch.randn((64,128), requires_grad=True)
            fake_data = G(z.to(device)).detach()  # [64, 1, 28, 28]
            real_data = image  # [64, 1, 28, 28]
            loss_d += train_discriminator(D_optimizer, real_data, fake_data)  # 판별자 학습
            
        z = torch.randn((batch_size, args['z_dim']), requires_grad=True)    
        fake_data = G(z.to(device))
        loss_g += train_generator(G_optimizer, fake_data)    # 생성자 학습
        
        generated_image = G(z).cpu().detach()
        generated_image = make_grid(generated_image)
        save_generator_image(generated_image, './assets/image_at_epoch_{:04d}.png')
        images.append(generated_image)
        
    epoch_loss_g = loss_g/idx  # 에포크에 대한 총 생성자 오차 계산
    epoch_loss_d = loss_d/idx  # 에포크에 대한 총 판별자 오차 계산
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    
    print(f"Epoch {epoch} of {args['num_epochs']}")
    print(f"Generator loss: {epoch_loss_g: .8f}, Discriminator loss: {epoch_loss_d: .8f}")
    
    
# %%
