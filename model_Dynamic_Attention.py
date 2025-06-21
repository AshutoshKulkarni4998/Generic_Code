import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class SCConv(nn.Module): #Self Calibrated Convolution
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(SCConv, self).__init__()

        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes*2, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes*2, kernel_size=3, stride=1,
                                padding=1, dilation=dilation,
                                groups=groups, bias=False),
                   
                    )

        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, 18, kernel_size=3, stride=stride,
                                padding=1, dilation=dilation,
                                groups=1, bias=False),
                   
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out
    
class Temporal_Alignment(torch.nn.Module): #Self-Calibrated Deformable Temporal Alignment Block
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU()):
        super(Temporal_Alignment, self).__init__()
        
        self.offset_conv1 = SCConv(in_channels*2, out_channels, 1, padding, dilation, 1, 2)
        self.deform1 =  DeformConv2d(in_channels, out_channels, 3, padding=1, groups=8)
        self.act_layer = act_layer
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def offset_gen(self, x, y):
        
        offset = torch.cat((x, y), dim=1)

        offset = self.offset_conv1(offset)
        mask = torch.sigmoid(offset)
        return offset, mask

    def forward(self, x, y):

        offset1,mask = self.offset_gen(x, y)
        feat1 = self.deform1(x, offset1, mask[:,:mask.shape[1]//2,:,:])

        # x = self.depthwise(feat)
        x = self.act_layer(feat1)
        
        x = self.pointwise(x)
        return x
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

     
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.reshape(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

       
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output



class DYNAMIC_ATTENTION(nn.Module):
    def __init__(self, channels, num_heads):
        super(DYNAMIC_ATTENTION, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1, bias=False)
        



        self.q_conv = Dynamic_conv2d(in_planes = channels, out_planes = channels, kernel_size = 3,
                  padding=1,  groups=channels, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class TransformerBlock_for_Video_Prompted(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock_for_Video_Prompted, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = DYNAMIC_ATTENTION(channels, num_heads)
        self.TA = Temporal_Alignment(channels, channels, 3)
        # self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, y):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = self.TA(x,y)
        # x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
        #                  .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels,channels_out):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels_out, kernel_size=3, padding=1, bias=False, stride = 2),
                                  )

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 2):
        super(UpSample, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x


class Video_Deraining(nn.Module):
    def __init__(self, num_heads=8, channels=16,
                 expansion_factor=2.66):
        super(Video_Deraining, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.embed_conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.transformer_block1 = TransformerBlock_for_Video_Prompted(channels, num_heads)
        self.transformer_block2 = TransformerBlock_for_Video_Prompted(channels*2, num_heads)
        self.transformer_block3 = TransformerBlock_for_Video_Prompted(channels*4, num_heads)
        self.transformer_block4 = TransformerBlock_for_Video_Prompted(channels*4, num_heads)
        self.transformer_block5 = TransformerBlock_for_Video_Prompted(channels*2, num_heads)
        self.transformer_block6 = TransformerBlock_for_Video_Prompted(channels, num_heads)
        self.transformer_block7 = TransformerBlock_for_Video_Prompted(channels, num_heads)

        self.downsample1 = DownSample(channels, channels*2)
        self.downsample2 = DownSample(channels*2, channels*4)
        self.downsample3 = DownSample(channels*4, channels*4)

        self.upsample1 = UpSample(channels*4, channels*2)
        self.upsample2 = UpSample(channels*6, channels)
        self.upsample3 = UpSample(channels*3, channels)
        
        self.recurrent_downsample1 = DownSample(channels, channels*2)
        self.recurrent_downsample2_1 = DownSample(channels, channels*2)
        self.recurrent_downsample2_2 = DownSample(channels*2, channels*4)
        self.recurrent_downsample3_1 = DownSample(channels, channels)
        self.recurrent_downsample3_2 = DownSample(channels, channels)
        self.recurrent_downsample3_3 = DownSample(channels, channels*4)
        self.recurrent_downsample4_1 = DownSample(channels, channels)
        self.recurrent_downsample4_2 = DownSample(channels, channels*2)
        self.recurrent_downsample5_1 = DownSample(channels, channels)


        self.output = nn.Conv2d(channels*2, 3, kernel_size=3, padding=1, bias=False)
        
        self.tan = nn.Tanh()

    def forward(self, x):
        x, recurrent_frame = x.chunk(2, dim=0)
        
        # print(x.shape, recurrent_frame.shape)
        fo = self.embed_conv(x)

        f1 = self.embed_conv1(recurrent_frame)
        
        transformer1 = self.transformer_block1(fo,f1)
        downsample_block1 = self.downsample1(transformer1)#128
        transformer2 = self.transformer_block2(downsample_block1, self.recurrent_downsample1(f1))
        downsample_block2 = self.downsample2(transformer2)#64
        transformer3 = self.transformer_block3(downsample_block2, self.recurrent_downsample2_2(self.recurrent_downsample2_1(f1)))
        downsample_block3 = self.downsample3(transformer3)#32
        transformer4 = self.transformer_block4(downsample_block3, self.recurrent_downsample3_3(self.recurrent_downsample3_2(self.recurrent_downsample3_1(f1))))
        upsample_block1 = self.upsample1(transformer4)#64

        transformer5 = self.transformer_block5(upsample_block1, self.recurrent_downsample4_2(self.recurrent_downsample4_1(f1)))
        concate1 = torch.cat([transformer5,transformer3], axis = 1)
        upsample_block2 = self.upsample2(concate1)#128
        transformer6 = self.transformer_block6(upsample_block2, self.recurrent_downsample5_1(f1))
        concate2 = torch.cat([transformer6,transformer2], axis = 1)
        upsample_block3 = self.upsample3(concate2)#256
        transformer7 = self.transformer_block7(upsample_block3, f1)
        concate3 = torch.cat([fo,transformer7], axis = 1)



        # print(offset1.shape)
        # exit(0)
        
        # x = self.depthwise(feat)
        out = self.tan(self.output(concate3)+x)
        return out




        



        
    
