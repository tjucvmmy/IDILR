import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import combinations, product
import math
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os


#Multi-Scale Residual Block(MSRB) 
class MSRB(nn.Module):
     def __init__(self,mid_channel):
        super(MSRB, self).__init__()
        
        self.conv_1=nn.Sequential(
             
                                  nn.AvgPool2d(kernel_size=1 ,stride=1),
                                  nn.Conv2d(mid_channel, mid_channel, 3, 1,1),
                                  nn.PReLU(),
                                 )
        
        self.conv_2=nn.Sequential(nn.AvgPool2d(kernel_size=2 ,stride=2),
                                  nn.Conv2d(mid_channel, mid_channel, 3, 1,1),
                                  nn.PReLU(),
                                 )
        
        self.conv_3=nn.Sequential(nn.AvgPool2d(kernel_size=4 ,stride=4),
                                  nn.Conv2d(mid_channel, mid_channel, 3,1, 1),
                                  nn.LeakyReLU(0.2)
                                 )
        
        self.cat = nn.Sequential(nn.Conv2d(3 * mid_channel, mid_channel, 1, 1),
                                 nn.PReLU(),
                                )

        self.res = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1), 
                                 nn.PReLU(),
                                 nn.Conv2d(mid_channel, mid_channel, 1, 1, 0), 
                                 
                                )
     def forward(self, x):
        b, c, h, w = x.size()

        conv1=self.conv_1(x)
        conv2=self.conv_2(x)
        conv3=self.conv_3(x)
        
        conv1=F.upsample_bilinear(conv1, size=[h, w])
        conv2=F.upsample_bilinear(conv2, size=[h, w])
        conv3=F.upsample_bilinear(conv3, size=[h, w])
        temp=torch.cat((conv1,conv2),dim=1)
        temp=torch.cat((temp,conv3),dim=1)
        cat=self.cat(temp)
        res = self.res(cat)
        out = res + x
        return out
##Multi-Channel Residual Block(MCRB) 
class MCRB(nn.Module):
     def __init__(self,mid_channel,mid_1_channel,mid_2_channel,mid_3_channel):
        super(MCRB, self).__init__()
        
        self.conv=nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 1, 1,0),
                                   nn.PReLU(),
                                  )
        
        
        self.conv_21=nn.Sequential(nn.Conv2d(mid_channel, mid_1_channel, 3, 1,1),
                                   nn.PReLU(),
                                  )
        self.pooling_1=nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.conv_22=nn.Sequential(nn.Conv2d(mid_channel, mid_2_channel, 5, 1,2),
                                   nn.PReLU(),
                                 )
        self.pooling_2=nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.conv_23=nn.Sequential(nn.Conv2d(mid_channel, mid_3_channel, 7,1, 3),
                                   nn.PReLU(),
                                 )
        self.pooling_3=nn.Sequential(nn.AdaptiveAvgPool2d(1))
        
    
        self.res = nn.Sequential(nn.Conv2d(mid_1_channel+mid_2_channel+mid_3_channel, mid_channel, 3, 1, 1), 
                                 nn.PReLU(),
                                 nn.Conv2d(mid_channel, mid_channel, 1, 1, 0), 
                               
                                )
     def forward(self, x):
       

        convert=self.conv(x)
        
        conv1=self.conv_21(convert)
        pooling_1=self.pooling_1(conv1)
        sigmoid_1=torch.sigmoid(pooling_1)
       
        conv1=torch.mul(sigmoid_1,conv1)

        conv2=self.conv_22(convert)
        pooling_2=self.pooling_2(conv2)
        sigmoid_2=torch.sigmoid(pooling_2)
        conv2=torch.mul(sigmoid_2,conv2)

        conv3=self.conv_23(convert)
        pooling_3=self.pooling_3(conv3)
        sigmoid_3=torch.sigmoid(pooling_3)
        conv3=torch.mul(sigmoid_3,conv3)
        
        temp=torch.cat((conv1,conv2),dim=1)
        temp=torch.cat((temp,conv3),dim=1)

        
        res=self.res(temp)
        out = res +x
        return out
   

class Derain_image(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Derain_image, self).__init__()
        self.convert = nn.Sequential(nn.Conv2d(in_channel, mid_channel, 3, 1, 1), 
                                     nn.PReLU(),
                                    )
        self.conv_a = nn.Sequential(MSRB(32))
        self.conv_b = nn.Sequential(MCRB(32,32,16,8))
        
        

        self.conv_c = nn.Sequential(MSRB(32))
        self.conv_d = nn.Sequential(MCRB(32,32,16,8))
        
        self.conv_e = nn.Sequential(MSRB(32))
        self.conv_f = nn.Sequential(MCRB(32,32,16,8))
        
        
        self.cat_1 = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 3, 1,1), nn.PReLU(),)
        self.cat_2 = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 3, 1,1), nn.PReLU(),)
        self.cat_3 = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 3, 1,1), nn.PReLU(),)
        self.cat_4 = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 3, 1,1),nn.PReLU(),)
        self.cat = nn.Sequential(nn.Conv2d(2*mid_channel, mid_channel, 3, 1,1), nn.PReLU(),)
       
        self.res=nn.Sequential(nn.Conv2d(mid_channel, out_channel, 3, 1,1), 
                                nn.PReLU(),
                                nn.Conv2d(out_channel,out_channel, 1, 1,0)
                               )
    def forward(self, x):
        
            convert = self.convert(x)
 
            
            s1 = self.conv_a(convert)
            c1 = self.conv_b(convert)
           

            temp1=torch.cat((s1,c1),dim=1)
            temp1=self.cat_1(temp1)
           
            

            s2 = self.conv_a(temp1)
            c2 = self.conv_b(temp1)
            
           
            temp2=torch.cat((s2,c2),dim=1)
            temp2=self.cat_2(temp2)
            
            

            s3  = self.conv_c(temp2)
            c3  = self.conv_d(temp2)
            
            
      
            s4  = self.conv_c(s3)
            c4  = self.conv_d(c3)
           
            temp3=torch.cat((s4,c4),dim=1)
            temp3=self.cat_3(temp3)

            s5 = self.conv_e(temp3)
            c5 = self.conv_f(temp3)
            
            temp4=torch.cat((s5,c5),dim=1)
            temp4=self.cat_4(temp4)

            s6 = self.conv_e(temp4)
            c6 = self.conv_f(temp4)
 
            temp=torch.cat((s6,c6),dim=1)
            temp=self.cat(temp)
            
            derain = self.res(temp)

            return derain 


class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.channel = 32
        self.conv_num = 4
        self.scale1 = nn.ModuleList()
        self.scale2 = nn.ModuleList()
        self.scale4 = nn.ModuleList()
        self.scale8 = nn.ModuleList()
        
        for i in range(self.conv_num):
            self.scale1.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale2.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale4.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
            self.scale8.append(nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2)))
        self.fusion84 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.fusion42 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.fusion21 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        self.pooling8 = nn.MaxPool2d(8, 8)
        self.pooling4 = nn.MaxPool2d(4, 4)
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.fusion_all = nn.Sequential(nn.Conv2d(4 * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        
    def forward(self, x):
      
        feature8 = self.pooling8(x)
        b8, c8, h8, w8 = feature8.size()
        feature4 = self.pooling4(x)
        b4, c4, h4, w4 = feature4.size()
        feature2 = self.pooling2(x)
        b2, c2, h2, w2 = feature2.size()
        feature1 = x
        b1, c1, h1, w1 = feature1.size()
        for i in range(self.conv_num):
            feature8 = self.scale8[i](feature8)
        scale8 = feature8
        feature4 = self.fusion84(torch.cat([feature4, F.upsample(scale8, [h4, w4])], dim=1))
        for i in range(self.conv_num):
            feature4 = self.scale4[i](feature4)
        scale4 = feature4
        feature2 = self.fusion42(torch.cat([feature2, F.upsample(scale4, [h2, w2])], dim=1))
        for i in range(self.conv_num):
            feature2 = self.scale2[i](feature2)

        scale2 = feature2
        feature1 = self.fusion21(torch.cat([feature1, F.upsample(scale2, [h1, w1])], dim=1))
        for i in range(self.conv_num):
            feature1 = self.scale1[i](feature1)
        scale1 = feature1
        fusion_all = self.fusion_all(torch.cat([scale1, F.upsample(scale2, [h1, w1]), F.upsample(scale4, [h1, w1]), F.upsample(scale8, [h1, w1])], dim=1))
        return fusion_all + x



class streak_Block(nn.Module):
    def __init__(self):
        super(streak_Block, self).__init__()
        self.channel=32
        self.convert = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, 1, 1),
            nn.PReLU(),
            )
        self.k1 = nn.Sequential(
                    nn.Conv2d(self.channel, self.channel, 3, 1, 1), 
                    nn.PReLU(),
                    )
        
        self.conv=nn.Sequential(nn.Conv2d(2*self.channel, self.channel, 3, 1, 1))
        self.relu = nn.PReLU()
    def forward(self,x):
        
        
        x2=self.convert(x)
        y2=self.k1(x2)
        y=torch.cat((x2,y2),dim=1)
        out=self.conv(y)
        out=self.relu(out+x)
        return out


class Rain_streak(nn.Module):
    def __init__(self,in_channel,mid_channel,out_channel):
        super(Rain_streak,self).__init__()
        
        self.conv=nn.Sequential(nn.Conv2d(in_channel,mid_channel,3,1,1))
       
        self.cat_1=nn.Sequential(nn.Conv2d(2*mid_channel,mid_channel,3,1,1),
                                 nn.PReLU(),
                                )
        self.cat_2=nn.Sequential(nn.Conv2d(3*mid_channel,mid_channel,3,1,1),
                                 nn.PReLU(),
                                )
        self.cat_3=nn.Sequential(nn.Conv2d(4*mid_channel,mid_channel,3,1,1),
                                 nn.PReLU(),
                                )
        self.cat_4=nn.Sequential(nn.Conv2d(5*mid_channel,mid_channel,3,1,1),
                                 nn.PReLU(),
                                )
        self.res = nn.Sequential(
                                 nn.Conv2d(mid_channel, out_channel, 3, 1, 1),
                                 nn.PReLU(),
                                 nn.Conv2d(out_channel,out_channel, 3, 1, 1),
                                )

        self.M1=nn.Sequential(streak_Block(),
                              streak_Block()
                             )
       
        self.M2=nn.Sequential(streak_Block(),
                              streak_Block()
                             )
       
        self.M3=nn.Sequential(streak_Block(),
                              streak_Block()
                             )
      
        self.M4=nn.Sequential(streak_Block(),
                              streak_Block()
                             )
        
      
    def forward(self,x):
        out_0=self.conv(x)
        
        
        y1=self.M1(out_0)
        temp_1=torch.cat((out_0,y1),dim=1)
        out_1=self.cat_1(temp_1)
        
        
        y2=self.M2(out_1)
        temp_2=torch.cat((temp_1,y2),dim=1)
        out_2=self.cat_2(temp_2)
        
        y3=self.M3(out_2)
        temp_3=torch.cat((temp_2,y3),dim=1)
        out_3=self.cat_3(temp_3)
        
        y4=self.M4(out_3)
        temp_4=torch.cat((temp_3,y4),dim=1)
        out_4=self.cat_4(temp_4)
       
        out=out_0+out_4
        

        streak=self.res(out)
        
        return streak

        
        
        
            
    
    
class  IDLIR(nn.Module):
    def __init__(self, recurrent_iter=5, use_GPU=True):
        super(IDLIR, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU
#-------------------------------------------------------
        self.image=nn.Sequential(Derain_image(6,32,3))
        self.streak=nn.Sequential(Rain_streak(3,32,3))
#------------------------------------------------------------
        self.fusion_O=nn.Sequential(fusion())
        self.fusion_B=nn.Sequential(fusion())
        self.fusion_R=nn.Sequential(fusion())
#--------------------------------------------------------
        self.convert=nn.Sequential(nn.Conv2d(6,3,3,1,1),
                                   nn.PReLU(),
                                   )
        
        self.cat_B=nn.Sequential(nn.Conv2d(6,32,3,1,1),
                                 nn.PReLU(),
                                )
        self.conv_B=nn.Sequential(nn.Conv2d(32,3,3,1,1),
                                  nn.PReLU(),
                                  nn.Conv2d(3,3,1,1,0),
                                )
        self.cat_R=nn.Sequential(nn.Conv2d(6,32,3,1,1),
                                 nn.PReLU(),
                                )
        self.conv_R=nn.Sequential(nn.Conv2d(32,3,3,1,1),
                                  nn.PReLU(),
                                  nn.Conv2d(3,3,1,1,0),
                                )
        self.cat_O=nn.Sequential(nn.Conv2d(6,32,3,1,1),
                                 nn.PReLU(),
                                )
        self.conv_O=nn.Sequential(nn.Conv2d(32,3,3,1,1),
                                  nn.PReLU(),
                                  nn.Conv2d(3,3,1,1,0),
                                )


    def forward(self, input):
         
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        
        O_difference= input
        B = Variable(torch.zeros(batch_size, 3, row, col))
        R = Variable(torch.zeros(batch_size, 3, row, col))
        
        if self.use_GPU:
            B = B.cuda()
            R = R.cuda()
           
       
       
        for i in range(self.iteration):
            
            R_difference=self.streak(O_difference)
            
            B_difference=self.image(O_difference)

            B=self.cat_B(torch.cat((B,B_difference),dim=1))
            B=self.fusion_B(B)
            B=self.conv_B(B)
            
            R=self.cat_R(torch.cat((R,R_difference),dim=1))
            R=self.fusion_R(R)
            R=self.conv_R(R)
            
      
            
            O=self.cat_O(torch.cat((B,R),dim=1))
            O=self.fusion_O(O)
            O=self.conv_O(O)
            
            
            O_difference=input-O

            
 
    
        return B, R
       
      
     


     
          
