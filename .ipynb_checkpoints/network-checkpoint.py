import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

'''
Inception-V4 implementation with 
1. all pooling kernel-size to be 3*3 and stride being 1 and padding being 1
2. Proper padding to maintain spatially same size as input
3. PReLU activation
4. reduced channel-size at each layer in order to fit local machine
'''
class conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(conv_Block , self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.batchNormalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.batchNormalization(out)
        out = self.activation(out)
        return out

class Stem_Block(nn.Module):
    def __init__(self , in_channels, initial_size):
        super(Stem_Block , self).__init__()
        self.conv1 = conv_Block(in_channels, initial_size, 3)
        self.conv2 = conv_Block(initial_size, initial_size, 3)
        self.conv3 = conv_Block(initial_size, initial_size * 2, 3)

        self.branch1 = conv_Block(initial_size * 2, initial_size * 3, 3)
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.branch1_1 = nn.Sequential(
            conv_Block(initial_size * 5, initial_size * 2, 1, padding=0),
            conv_Block(initial_size * 2, initial_size * 2, (1,7), padding=(0,3)),
            conv_Block(initial_size * 2, initial_size * 2, (7,1), padding=(3,0)),
            conv_Block(initial_size * 2, initial_size * 3, 3)
        )
        self.branch2_1 = nn.Sequential(
            conv_Block(initial_size * 5, initial_size * 2, 1, padding=0),
            conv_Block(initial_size * 2, initial_size * 3, 3)
        )

        self.branch1_2 = conv_Block(initial_size * 6, initial_size * 6, 3)

        self.branch2_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        branch1 = self.branch1(out)
        branch2 = self.branch2(out)
        out = torch.cat((branch1 , branch2), axis=1)

        branch1 = self.branch1_1(out)
        branch2 = self.branch2_1(out)
        out = torch.cat((branch1 , branch2), 1)  

        branch1 = self.branch1_2(out)
        branch2 = self.branch2_2(out)
        out = torch.cat((branch1 , branch2), axis=1)
        return out

class inception_Block_A(nn.Module):
    def __init__(self, in_channels, initial_size):
        super(inception_Block_A , self).__init__()    

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, initial_size * 2, 1, padding=0),
            conv_Block(initial_size * 2, initial_size * 3, 3),
            conv_Block(initial_size * 3, initial_size * 3, 3)
        )
        self.branch2 = nn.Sequential(
            conv_Block(in_channels, initial_size * 2, 1, padding=0),
            conv_Block(initial_size * 2, initial_size * 3, 3)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_Block(in_channels, initial_size * 3, 1, padding=0)
        )
        self.branch4 = conv_Block(in_channels, initial_size * 3, 1, padding=0)
    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), axis=1)
        return out

class inception_Block_B(nn.Module):
    def __init__(self, in_channels, initial_size):
        super(inception_Block_B, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, initial_size * 6, 1, padding=0),
            conv_Block(initial_size * 6, initial_size * 6, (7,1), padding=(3,0)),
            conv_Block(initial_size * 6, initial_size * 7, (1,7), padding=(0,3)),
            conv_Block(initial_size * 7, initial_size * 7, (7,1), padding=(3,0)),
            conv_Block(initial_size * 7, initial_size * 8, (1,7), padding=(0,3)),
        )

        self.branch2 = nn.Sequential(
            conv_Block(in_channels, initial_size * 6, 1, padding=0),
            conv_Block(initial_size * 6, initial_size * 7, (1,7), padding=(0,3)),
            conv_Block(initial_size * 7, initial_size * 8, (7,1), padding=(3,0)),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_Block(in_channels, initial_size * 4, 1, padding=0)   
        )

        self.branch4 = conv_Block(in_channels, initial_size * 12, 1, padding=0)
        
    def forward(self , x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)  

        out = torch.cat((branch1, branch2, branch3, branch4), axis=1)

        return out


class inception_Block_C(nn.Module):
    def __init__(self, in_channels, initial_size):
        super(inception_Block_C , self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, initial_size * 12, 1, padding=0),
            conv_Block(initial_size * 12, initial_size * 14, (3,1), padding=(1,0)),
            conv_Block(initial_size * 14, initial_size * 16, (1,3), padding=(0,1))
        )  

        self.branch1_1 = conv_Block(initial_size * 16, initial_size * 8, (1,3), padding=(0,1))
        self.branch1_2 = conv_Block(initial_size * 16, initial_size * 8, (3,1), padding=(1,0))

        self.branch2 = conv_Block(in_channels, initial_size * 12, 1, padding=0)

        self.branch2_1 = conv_Block(initial_size * 12, initial_size * 8, (1,3), padding=(0,1))
        self.branch2_2 = conv_Block(initial_size * 12, initial_size * 8, (3,1), padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1 ,padding=1),
            conv_Block(in_channels, initial_size * 8, 3)
        )

        self.branch4 = conv_Block(in_channels, initial_size * 8, 1, padding=0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch1_1 = self.branch1_1(branch1)
        branch1_2 = self.branch1_2(branch1)  
        branch1 = torch.cat((branch1_1, branch1_2), axis=1)

        branch2 = self.branch2(x)
        branch2_1 = self.branch2_1(branch2)
        branch2_2 = self.branch2_2(branch2)
        branch2 = torch.cat((branch2_1, branch2_2), axis=1)

        branch3 = self.branch3(x)

        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), axis=1)

        return out

class reduction_Block_A(nn.Module):
    def __init__(self, in_channels, initial_size):
        super(reduction_Block_A, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, initial_size * 6, 1, padding=0),
            conv_Block(initial_size * 6, initial_size * 7, 3),
            conv_Block(initial_size * 7, initial_size * 8, 3, stride=1, padding=1)
        )

        # self.branch2 = conv_Block(in_channels , 384 , 3 , 2 , 0)
        self.branch2 = conv_Block(in_channels, initial_size * 12, 3)
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat((branch1 ,branch2 , branch3), axis=1)

        return out

class reduction_Block_B(nn.Module):
    def __init__(self, in_channels, initial_size):
        super(reduction_Block_B, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, initial_size * 8, 1, padding=0),
            conv_Block(initial_size * 8, initial_size * 8, (1,7), padding=(0,3)),
            conv_Block(initial_size * 8, initial_size * 10, (7,1), padding=(3,0)),
            conv_Block(initial_size * 10, initial_size * 10, 3, stride=1, padding=1)
            # conv_Block(320 ,320 ,3 ,2 , 0)  
        )

        self.branch2 = nn.Sequential(
            conv_Block(in_channels, initial_size * 6, 1, padding=0),
            conv_Block(initial_size * 6, initial_size * 6, 3, stride=1, padding=1)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self , x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat((branch1, branch2, branch3), axis=1)

        return out
    

class InceptionV4(nn.Module):
    def __init__(self, initial_size):
        super(InceptionV4 , self).__init__()

        self.stem = Stem_Block(3, initial_size)

        self.inceptionA = inception_Block_A(initial_size * 12, initial_size)

        self.reductionA = reduction_Block_A(initial_size * 12, initial_size)

        self.inceptionB = inception_Block_B(initial_size * 32, initial_size)

        self.reductionB = reduction_Block_B(initial_size * 32, initial_size)

        self.inceptionC = inception_Block_C(initial_size * 48, initial_size)

    def forward(self,x):

        out = self.stem(x)

        out = self.inceptionA(out)
        out = self.inceptionA(out)
        out = self.inceptionA(out)
        out = self.inceptionA(out)

        out = self.reductionA(out)


        out = self.inceptionB(out)
        out = self.inceptionB(out)
        out = self.inceptionB(out)
        out = self.inceptionB(out)
        out = self.inceptionB(out)
        out = self.inceptionB(out)
        out = self.inceptionB(out)

        out = self.reductionB(out)

        out = self.inceptionC(out)
        out = self.inceptionC(out)
        out = self.inceptionC(out)

        return out
    
'''
Astrous convolution 
'''
class atrous_Conv(nn.Module):
    def __init__(self, inplanes, rate, num_classes=3):
        super(atrous_Conv, self).__init__()
        planes = inplanes
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        # self.fc1 = conv_Block(planes, planes, 1)
        # self.fc1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)
        # self.fc2 = nn.Conv2d(planes, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        return x
    
'''
Pyramid max pooling with convolution size 1, 3, 5, 7 and element-wise max operation
'''
class m_Beta_Model(nn.Module):
    def __init__(self, in_feature, out_channel, normalize=False):
        super(m_Beta_Model, self).__init__()
        self.M_beta_1 = nn.Conv2d(in_feature, out_channel, kernel_size=1, stride=1, padding=0)
        self.M_beta_3 = nn.Conv2d(in_feature, out_channel, kernel_size=3, stride=1, padding=1)
        self.M_beta_5 = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size=(1,5), stride=1, padding=(0,2)),
            nn.PReLU(),
            nn.Conv2d(in_feature, out_channel, kernel_size=(5,1), stride=1, padding=(2,0))
        )
        self.M_beta_7 = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.PReLU(),
            nn.Conv2d(in_feature, out_channel, kernel_size=(7,1), stride=1, padding=(3,0))
        )
    
    def forward(self, x):
        out = self.M_beta_1(x)
        out = torch.maximum(out, self.M_beta_3(x))
        out = torch.maximum(out, self.M_beta_5(x))
        out = torch.maximum(out, self.M_beta_7(x))
        return out

'''
Pyramid sum with convolution size 1, 3, 5, 7 and element-wise sum operation
'''
class m_Beta_Model_sum(nn.Module):
    def __init__(self, in_feature, out_channel, normalize=False):
        super(m_Beta_Model_sum, self).__init__()
        self.M_beta_1 = nn.Conv2d(in_feature, out_channel, kernel_size=1, stride=1, padding=0)
        self.M_beta_3 = nn.Conv2d(in_feature, out_channel, kernel_size=3, stride=1, padding=1)
        self.M_beta_5 = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size=(1,5), stride=1, padding=(0,2)),
            nn.PReLU(),
            nn.Conv2d(in_feature, out_channel, kernel_size=(5,1), stride=1, padding=(2,0))
        )
        self.M_beta_7 = nn.Sequential(
            nn.Conv2d(in_feature, in_feature, kernel_size=(1,7), stride=1, padding=(0,3)),
            nn.PReLU(),
            nn.Conv2d(in_feature, out_channel, kernel_size=(7,1), stride=1, padding=(3,0))
        )
    
    def forward(self, x):
        out = self.M_beta_1(x)
        out = out + self.M_beta_3(x)
        out = out + self.M_beta_5(x)
        out = out + self.M_beta_7(x)
        return out
    
    
'''
DesnowNet implementation with two inception-v4 backbone defined as Dt and Dr. 
With astrous pooling for feature extraction
With pyramid sum or max operation to implement Rt and Rg block
in_feature is 1536 for original size inception and 768 for current reduced size version
'''
class Snow_model(nn.Module):
    def __init__(self, gamma=4, initial_size=16):
        super(Snow_model, self).__init__()
        in_feature = initial_size * 48
        self.backbone_dt = InceptionV4(initial_size)
        self.backbone_dr = InceptionV4(initial_size)
        self.gamma = gamma
        self.aspps = nn.ModuleList()
        for i in range(gamma):
            self.aspps.append(atrous_Conv(in_feature, 2**i))
            
        self.aspps_dr = nn.ModuleList()
        for i in range(gamma):
            self.aspps_dr.append(atrous_Conv(in_feature, 2**i))
            
        self.AE = m_Beta_Model(gamma * in_feature, 3)
        self.SE = m_Beta_Model(gamma * in_feature, 1)
        self.RG = m_Beta_Model_sum(gamma * in_feature, 3)
        
    def forward(self, x, train=True, viz=False):
        phi = self.backbone_dt(x)
        concat = []
        for i in range(self.gamma):
            concat.append(self.aspps[i](phi))
        out = torch.cat(concat, axis=1)
        
        a = self.AE(out)
        z_hat = torch.clamp(self.SE(out), min=0, max=1)
        
        mask = torch.where(z_hat==1, 1, 0)
        y_prime = mask * x + mask.logical_not() * ((x - a * mask.logical_not() * z_hat) / (1 - mask.logical_not() * z_hat))
        
        
        fc = y_prime * z_hat * a
        
        phi_dr = self.backbone_dr(fc)
        concat_dr = []
        for i in range(self.gamma):
            concat_dr.append(self.aspps_dr[i](phi_dr))
        out_dr = torch.cat(concat_dr, axis=1)
        r = self.RG(out_dr)
        
        y_hat = y_prime + r
        
        '''
        Crop y_hat at inference time to [0, 1] as suggested by the paper
        '''
        if not train:
            y_hat = torch.clamp(y_hat, min=0, max=1)
        if viz:
            return y_hat, y_prime, z_hat, r, a, fc
        else:
            return y_hat, y_prime, z_hat
        
        
'''
Loss pyramid to check non-local spatial correctness in loss function. 
Parameter chosen to be tao=4 since all other pyramid having 4-level of depth
'''
class loss_pyramid(nn.Module):
    def __init__(self, tao):
        super(loss_pyramid, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=tao, stride=tao)
    def forward(self, x):
        return self.maxpool(x)
        

'''
Loss function with pyramid structure and weight-regularization.
'''
class Snow_loss(nn.Module):
    def __init__(self, lambda_zhat=3, tao=4, lambda_weight_reg=5e-4):
        super(Snow_loss, self).__init__()  
        self.lambda_zhat = lambda_zhat
        self.tao = tao
        self.lambda_weight_reg = lambda_weight_reg
        self.loss_yhat = nn.ModuleList()
        for i in range(tao):
            self.loss_yhat.append(loss_pyramid(2**i))
        self.loss_yprime = nn.ModuleList()
        for i in range(tao):
            self.loss_yprime.append(loss_pyramid(2**i))
        self.loss_zhat = nn.ModuleList()
        for i in range(tao):
            self.loss_zhat.append(loss_pyramid(2**i))
        
    def forward(self, y_hat, y_prime, z_hat, y, z, weight_reg):
        loss_yhat = nn.functional.mse_loss(self.loss_yhat[0](y_hat), self.loss_yhat[0](y))
        for i in range(1, self.tao):
            loss_yhat += nn.functional.mse_loss(self.loss_yhat[i](y_hat), self.loss_yhat[i](y))
            
        loss_yprime = nn.functional.mse_loss(self.loss_yprime[0](y_prime), self.loss_yprime[0](y))
        for i in range(1, self.tao):
            loss_yprime += nn.functional.mse_loss(self.loss_yprime[i](y_prime), self.loss_yprime[i](y))
        
        loss_zhat = nn.functional.mse_loss(self.loss_zhat[0](z_hat), self.loss_zhat[0](z))
        for i in range(1, self.tao):
            loss_zhat += nn.functional.mse_loss(self.loss_zhat[i](z_hat), self.loss_zhat[i](z))
        
        return loss_yhat + loss_yprime + self.lambda_zhat * loss_zhat + self.lambda_weight_reg * weight_reg
    
    

'''
Snow dataset with random croping the image to be 3*64*64 as suggested in the paper
'''
class Snow_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True):
        if isinstance(data_path, str):
            self.data_path = data_path
            if train:
                self.data = np.load(f"{data_path}/train_data.npy", allow_pickle=True)
            else:
                self.data = np.load(f"{data_path}/validate_data.npy", allow_pickle=True)
            self.crop = transforms.RandomCrop(64)
            self.totensor = transforms.ToTensor()
        else:
            raise ValueError('not supported data path format')        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        z = Image.open(f"{self.data_path}/mask/{self.data[idx]}")
        y = Image.open(f"{self.data_path}/gt/{self.data[idx]}")
        x = Image.open(f"{self.data_path}/synthetic/{self.data[idx]}")
        z = transforms.Grayscale()(z)
        i, j, h, w = self.crop.get_params(z, output_size=(64, 64))
        z = self.totensor(transforms.functional.crop(z, i, j, h, w))
        y = self.totensor(transforms.functional.crop(y, i, j, h, w))
        x = self.totensor(transforms.functional.crop(x, i, j, h, w))

        return z, y, x

    def dataset_len(self):
        return len(self.data)
    
    
