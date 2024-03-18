from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import multiprocessing
from multiprocessing import Process, freeze_support

def run():
    freeze_support()
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataroot = "Pytorch\Image\celeba"
    workers = 0
    batch_size = 128
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1

    ## 데이터셋 불러오기
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    ## 가중치 초기화(DCGAN은 mean=0, stdev=0.02인 정규분포 사용하여 무작위 초기화 하는 것이 좋다고 함)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    ########################################################################
    ## 생성자: 잠재 공간 벡터 z를 데이터 공간으로 변환
    ## -> 학습이미지와 같은 사이즈를 가진 RGB 이미지를 생성     
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # 입력데이터 z가 가장 처음 통과하는 전치 Convolutional layer
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                
                # 위의 계층을 통과한 데이터의 크기. '(ngf*8) * 4 * 4'
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                # 위의 계층을 통과한 데이터의 크기. '(ngf*4) * 8 * 8'
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                # 위의 계층을 통과한 데이터의 크기. '(ngf*2) * 16 * 16'
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),


                # 위의 계층을 통과한 데이터의 크기. '(ngf) * 32 * 32'
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                ## 출력값을 [-1, 1] 사이의 범위로 조정
                nn.Tanh()
                
                # 위의 계층을 통과한 데이터의 크기 '(nc) * 64 * 64'
            )
        def forward(self, input):
            return self.main(input)

    # 생성자 만들기
    netG = Generator(ngpu).to(device)

    if(device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # 모든 가중치 평균, 분산 초기화
    netG.apply(weights_init)

    print(netG)
    ########################################################################
    # 구분자 -> 입력 이미지 진짜 or 가짜 판별
    # 3*64*64 이미지 입력받아, Conv2d - BatchNorm2d - LeackReLU 게층 통과
    # 마지막 출력에서 sigmoid 함수 이용하여 0~! 사이의 확률값으로 조정
    # 보폭이 있는 strided 합성곱을 사용하면 신경망 내에서 스스로의 pooling 함수를 학습
    # -> 직접적으로 풀링 계층 사용보다 더 유리
    # -> Leacky ReLU 함수는 학습과정에서 G와 D가 더 효과적인 Gradient 얻을 수 있음

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # 입력 데이터의 크기는 ``(nc) x 64 x 64`` 입니다
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. ``(ndf) x 32 x 32``
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. ``(ndf*2) x 16 x 16``
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. ``(ndf*4) x 8 x 8``
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. ``(ndf*8) x 4 x 4``
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)
        
    netD = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)
    print(netD)
    ########################################################################

    ## Loss function and Optimizer
    criterion = nn.BCELoss()
    # 생성자의 학습상태 확인할 잠재 공간 벡터
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # 학습에 사용되는 참/거짓 라벨
    real_label = 1.
    fake_label = 0.

    # G와 D에서 사용할 adam optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    ########################################################################
    # 학습
    # part 1: 구분자의 학습 -> Gradient를 상승시키며 훈련 -> log(D(x)) + log(1-D(G(z))) 최대화
    # part 2: 생성자의 학습 -> log(D(G(z))) 최대화

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # def main():
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
            ###########################
            ## 진짜 데이터들로 학습을 합니다
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## 가짜 데이터들로 학습
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # G를 이용해 가짜 이미지 생성
            fake = netG(noise)
            label.fill_(fake_label)
            # D를 이용해 데이터의 진위 판별
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            # 역판를 통해 변화도 계산 -> 앞서 구한 변화도에 accumulate
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값 더하기
            # errD는 역전파에서 사용되지 않고, 이후 학습 상태 reporting 시 사용
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            # 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터 통과
            # 이때 G는 업데이트되지 않지만, D가 업데이트 되기 때문에 앞선 손실값이 다른 값이 나옴
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # 훈련 상태 출력
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # 손실값 저장
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # fixed noise 통과시킨 G의 출력값 저장
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    ## 학습하는 동안의 손실값들
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    ## G의 학습 과정 시각화
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    ## 진짜 이미지 vs 가짜 이미지
    real_batch = next(iter(dataloader))

    # 진짜 이미지 화면에 출력
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # 가짜 이미지들을 화면에 출력합니다
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

if __name__ == '__main__':
    run()
#     freeze_support()
#     main()
#     weights_init()
#     Generator.__init__()
#     Generator.forward()
#     Discriminator.__init__()
#     Discriminator.forward()