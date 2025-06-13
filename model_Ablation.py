import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision
import matplotlib.pyplot as plt
import numpy as np
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out
def visualize_featuremaps_batchwise(ylh, filename):
  processed = []

  for i, feature_map in enumerate(ylh):

      feature_map = feature_map.squeeze(0)
      for j in range(feature_map.shape[0]):

        gray_scale = feature_map[j]
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
        plt.imsave(filename + '_' +  str(j) + '.png', processed[j])

  # fig = plt.figure(figsize=(10, 20))
  # for i in range(len(processed)):
  #   a = fig.add_subplot(1, 2, i+1)
  #   imgplot = plt.imshow(processed[i])
  #   a.axis("off")
  #     # a.set_title(names[i].split('(')[0], fontsize=30)
  # plt.savefig(filename, bbox_inches='tight')

  
  # plt.imshow(processed[0],cmap='Reds')
class DMB(nn.Module):
    def __init__(self, channels, num_heads):
        super(DMB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.deg = nn.Conv2d(channels, channels * 3, kernel_size=3, padding=1, bias=False)
        self.deg_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)
        self.sigmoid_q = nn.Sigmoid()
        self.sigmoid_k = nn.Sigmoid()
        self.sigmoid_v = nn.Sigmoid()


    def forward(self, x, degraded_features):

        b, c, h, w = x.shape

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        

        
        deg_feat_q, deg_feat_k, deg_feat_v = self.deg_conv(self.deg(x)).chunk(3, dim=1)


        q = torch.cat([q, deg_feat_q], dim=1)

        # k = torch.cat([k, deg_feat_k], dim=1)
        # v = torch.cat([v, deg_feat_v], dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)


        
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
       

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)

        self.conv1_1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1,
                              groups=hidden_channels, bias=False)

        self.conv3_3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size= 3, padding=1,
                              groups=hidden_channels, bias=False)


        self.conv5_5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size= 5, padding=2,
                              groups=hidden_channels, bias=False)






        
        self.project_out = nn.Conv2d(hidden_channels*2, channels, kernel_size=1, bias=False)
        


    def forward(self, x):
        x = self.project_in(x)
        x1 = self.conv1_1(x)
        x3 = self.conv3_3(x)
        x5 = self.conv5_5(x)

        gate1 = F.gelu(x1)*x3
        gate2 = F.gelu(x1)*x5

        # gate_only_1x1 = F.gelu(x1)*x1
        # visualize_featuremaps_batchwise(gate_only_1x1, "gate_only_1x1.png")
        # gate_only_3x3 = F.gelu(x3)*x3
        # visualize_featuremaps_batchwise(gate_only_3x3, "gate_only_3x3.png")
        # gate_only_5x5 = F.gelu(x5)*x5
        # visualize_featuremaps_batchwise(gate_only_5x5, "gate_only_5x5.png")



        x = self.project_out(torch.cat([gate1,gate2],axis = 1))
        # visualize_featuremaps_batchwise(x, "actual.png")
        # x1 =  self.project_out((torch.cat([gate_only_1x1,gate_only_1x1],axis = 1)))
        # visualize_featuremaps_batchwise(x1, "x1gate_only_1x1.png")
        # x2 =  self.project_out((torch.cat([gate_only_3x3,gate_only_3x3],axis = 1)))
        # visualize_featuremaps_batchwise(x2, "x2gate_only_3x3.png")

        # x3 = self.project_in(gate_only_5x5)
        # x3 =  self.project_out((torch.cat([gate_only_5x5,gate_only_5x5],axis = 1)))
        # visualize_featuremaps_batchwise(x3, "x3gate_only_5x5.png")



        # x1, x3 = self.conv1(self.project_in(x)).chunk(2, dim=1)
        # x2, x4 = self.conv2(self.project_in(x)).chunk(2, dim=1)



        # x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class TransformerBlock_Degradation_Aware(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock_Degradation_Aware, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = DMB(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.norm_degraded = nn.LayerNorm(16)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, degraded_features):
        b, c, h, w = x.shape
        b1,c1,h1,w1 = degraded_features.shape

        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w), self.norm_degraded(degraded_features.reshape(b1, c1, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b1, c1, h1, w1))

        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x

class DownSample(nn.Module):
    def __init__(self, channels,channels_out):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels_out, kernel_size=3, padding=1, bias=False, stride = 2),
                                  )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels,channels_out):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.ConvTranspose2d(channels, channels_out, kernel_size=2, bias=False, stride = 2),
                                 )

    def forward(self, x):
        return self.body(x)

class GAFP(nn.Module):
    def __init__(self, channels1,channels2, channels_out, m=-0.80):
        super(GAFP, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

        self.project_conv1 = nn.Conv2d(channels1, channels_out//2, kernel_size=1, bias=False)
        self.project_conv2 = nn.Conv2d(channels2, channels_out//2, kernel_size=1, bias=False)

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        # print("MIX VALUE IS", mix_factor)
        conv1 = self.project_conv1(fea1)
        conv2 = self.project_conv2(fea2)
        conv1_mixed = conv1*mix_factor.expand_as(conv1)
        conv2_mixed = conv2*(1-mix_factor.expand_as(conv2))
        gel1 = F.gelu(conv1_mixed)
        gel2 = F.gelu(conv2_mixed)
        gate_mult1 = gel1*conv1_mixed
        gate_mult2 = gel2*conv2_mixed

        out = torch.cat([gate_mult1, gate_mult2], dim=1)
        return out

class AFM(nn.Module):
    def __init__(self, channels1,channels2, channels_out, m=-0.80):
        super(AFM, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

        self.project_conv1 = nn.Conv2d(channels1, channels_out//2, kernel_size=1, bias=False)
        self.project_conv2 = nn.Conv2d(channels2, channels_out//2, kernel_size=1, bias=False)

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        # print("MIX VALUE IS", mix_factor)
        conv1 = self.project_conv1(fea1)
        conv2 = self.project_conv2(fea2)
        conv1_mixed = conv1*mix_factor.expand_as(conv1)
        conv2_mixed = conv2*(1-mix_factor.expand_as(conv2))
        # gel1 = F.gelu(conv1_mixed)
        # gel2 = F.gelu(conv2_mixed)
        # gate_mult1 = gel1*conv1_mixed
        # gate_mult2 = gel2*conv2_mixed

        out = torch.cat([conv1_mixed, conv2_mixed], dim=1)
        return out
class Restormer(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[32, 64, 128, 256], num_refinement=2,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)

        self.embed_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.embed_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.embed_conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.embed_conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.embed_conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)

        
        self.upper_block_01 = TransformerBlock(16, 8, expansion_factor)
        self.upper_block_02 = TransformerBlock(16, 8, expansion_factor)
        self.upper_block_03 = TransformerBlock(16, 8, expansion_factor)
        self.upper_block_04 = TransformerBlock(16, 8, expansion_factor)
        self.upper_block_05 = TransformerBlock(16, 8, expansion_factor)
        self.upper_block_06 = TransformerBlock(16, 8, expansion_factor)

        self.encoder_block_01 = TransformerBlock(16, 1, expansion_factor)
        

        self.encoder_block_02 = TransformerBlock(32, 2, expansion_factor)
        # self.encoder_block_02_1 = TransformerBlock(32, 2, expansion_factor)
        

        self.encoder_block_03 = TransformerBlock(64, 4, expansion_factor)
        # self.encoder_block_03_1 = TransformerBlock(64, 4, expansion_factor)
        # self.encoder_block_03_2 = TransformerBlock(64, 4, expansion_factor)


        self.encoder_block_04 = TransformerBlock(128, 8, expansion_factor)
        # self.encoder_block_04_1 = TransformerBlock(128, 8, expansion_factor)
        # self.encoder_block_04_2 = TransformerBlock(128, 8, expansion_factor)
        # self.encoder_block_04_3 = TransformerBlock(128, 8, expansion_factor)


        self.encoder_block_05 = TransformerBlock(64, 4, expansion_factor)
        # self.encoder_block_05_1 = TransformerBlock(64, 4, expansion_factor)
        # self.encoder_block_05_2 = TransformerBlock(64, 4, expansion_factor)
        # 

        self.encoder_block_06 = TransformerBlock(32, 2, expansion_factor)
        # self.encoder_block_06_1 = TransformerBlock(32, 2, expansion_factor)


        self.encoder_block_07 = TransformerBlock(16, 1, expansion_factor)


        self.upper_downs1 = DownSample(16, 16)
        self.upper_downs2 = DownSample(16, 16)
        self.upper_downs3 = DownSample(16, 16)
        self.upper_downs4 = DownSample(16, 16)
        self.upper_downs5 = DownSample(16, 16)
        self.upper_downs6 = DownSample(16, 16)
        self.upper_downs7 = DownSample(16, 16)
        self.upper_downs8 = DownSample(16, 16)
        self.upper_downs9 = DownSample(16, 16)



        self.degradation_downsample_1 = DownSample(16, 16)

        self.degradation_downsample_2_1 = DownSample(16, 16)
        self.degradation_downsample_2_2 = DownSample(16, 16)

        self.degradation_downsample_3_1 = DownSample(16, 16)
        self.degradation_downsample_3_2 = DownSample(16, 16)
        self.degradation_downsample_3_3 = DownSample(16, 16)

        self.degradation_downsample_4_1 = DownSample(16, 16)
        self.degradation_downsample_4_2 = DownSample(16, 16)

        self.degradation_downsample_5_1 = DownSample(16, 16)
       



        self.enc_downs1 = DownSample(16, 32)
        self.enc_downs2 = DownSample(64, 64)
        self.enc_downs3 = DownSample(128, 128)
        self.enc_ups1 = UpSample(64, 64)
        self.enc_ups2 = UpSample(32, 32)
        self.enc_ups3 = UpSample(16, 16)

        self.feature_propagator_1 = GAFP(16, 32, 64, m = -1)
        self.feature_propagator_2 = GAFP(16, 64, 128, m = -0.8)
        self.feature_propagator_3 = GAFP(16, 128, 64, m = -0.6)
        self.feature_propagator_4 = GAFP(16, 64, 32, m = -0.4)
        self.feature_propagator_5 = GAFP(16, 32, 16, m = -0.2)


        # self.enc_downs = DownSample(16)

        # self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
        #     num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
        #                                zip(num_blocks, num_heads, channels)])
        # # the number of down sample or up sample == the number of encoder - 1
        # self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        # self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # # the number of reduce block == the number of decoder - 1
        # self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
        #                               for i in reversed(range(2, len(channels)))])
        # # the number of decoder == the number of encoder - 1
        # self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
        #                                                for _ in range(num_blocks[2])])])
        # self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
        #                                      for _ in range(num_blocks[1])]))
        # # the channel of last one is not change
        # self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
        #                                      for _ in range(num_blocks[0])]))

        # self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
        #                                   for _ in range(num_refinement)])
        self.output = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        # visualize_featuremaps_batchwise(fo, "fo_Thick")
        DAB1 = self.embed_conv1(fo)
        # visualize_featuremaps_batchwise(DAB1, "DAB1_Thick")
        DAB2 = self.embed_conv2(DAB1)
        # visualize_featuremaps_batchwise(DAB2, "DAB2_Thick")
        DAB3 = self.embed_conv3(DAB2)
        # visualize_featuremaps_batchwise(DAB3, "DAB3_Thick")
        DAB4 = self.embed_conv4(DAB3)
        # visualize_featuremaps_batchwise(DAB4, "DAB4_Thick")
        DAB5 = self.embed_conv5(DAB4)
        
        # visualize_featuremaps_batchwise(DAB5, "self.embed_conv5.png")
        # exit(0)

        out_upper1 = self.upper_block_01(fo, DAB5)




        
        # print(tmp.shape)  
        # exit(0)
        # print("Out_upper1 ::::  ", out_upper1.shape)
        out_upper_down1 = self.upper_downs1(out_upper1)
        # print("out_upper_down1 ::::  ", out_upper_down1.shape)


        out_upper2 = self.upper_block_02(out_upper1, DAB5)
        # print("out_upper2 ::::  ", out_upper2.shape)
        out_upper_down2 = self.upper_downs2(out_upper2)
        # print("out_upper_down2 ::::  ", out_upper_down2.shape)

        out_upper_down3 = self.upper_downs3(out_upper_down2)
        # print("out_upper_down3 ::::  ", out_upper_down3.shape)
        out_upper3 = self.upper_block_03(out_upper2, DAB5)
        # print("out_upper3 ::::  ", out_upper3.shape)
        out_upper_down4 = self.upper_downs4(out_upper3)
        # print("out_upper_down4 ::::  ", out_upper_down4.shape)
        out_upper_down5 = self.upper_downs5(out_upper_down4)
        # print("out_upper_down5 ::::  ", out_upper_down5.shape)
        out_upper_down6 = self.upper_downs6(out_upper_down5)
        # print("out_upper_down6 ::::  ", out_upper_down6.shape)
        
        out_upper4 = self.upper_block_04(out_upper3, DAB5)
        # print("out_upper4 ::::  ", out_upper4.shape)
        out_upper_down7 = self.upper_downs5(out_upper4)
        # print("out_upper_down7 ::::  ", out_upper_down7.shape)
        out_upper_down8 = self.upper_downs6(out_upper_down7)    
        # print("out_upper_down8 ::::  ", out_upper_down8.shape)
       
        out_upper5 = self.upper_block_05(out_upper4, DAB5)
        # print("out_upper5 ::::  ", out_upper5.shape)

        out_upper_down9 = self.upper_downs4(out_upper5)
        # print("out_upper_down9 ::::  ", out_upper_down9.shape)

        out_upper6 = self.upper_block_06(out_upper5, DAB5)
        # print("out_upper6 ::::  ", out_upper6.shape)
        '''=========================================================================================================='''
        '''=========================================================================================================='''
        out_enc1 = self.encoder_block_01(fo, DAB5)

        # print("out_enc1 ::::  ", out_enc1.shape)
        down1    = self.enc_downs1(out_enc1)
        # print("down1 ::::  ", down1.shape)
        out_enc2 = self.encoder_block_02(down1)
        # out_enc2_1 = self.encoder_block_02_1(out_enc2)
        # print("out_enc2 ::::  ", out_enc2.shape)
        feature_prop_1 = self.feature_propagator_1(out_upper_down1, out_enc2) 
        # exit(0)
        # visualize_featuremaps_batchwise(feature_prop_1, "feature_prop_1.png")
        # print("feature_prop_1 ::::  ", feature_prop_1.shape)
        down2    = self.enc_downs2(feature_prop_1)
        # print("down2 ::::  ", down2.shape)
        out_enc3 = self.encoder_block_03(down2)
        # out_enc3_1 = self.encoder_block_03_1(out_enc3)
        # out_enc3_2 = self.encoder_block_03_2(out_enc3_1)
        # print("out_enc3 ::::  ", out_enc3.shape)
        feature_prop_2 = self.feature_propagator_2(out_upper_down3, out_enc3)
        
        
        # print("feature_prop_2 ::::  ", feature_prop_2.shape)
        down3    = self.enc_downs3(feature_prop_2)
        # print("down3 ::::  ", down3.shape)

        out_enc4 = self.encoder_block_04(down3)
        # out_enc4_1 = self.encoder_block_04_1(out_enc4)
        # out_enc4_2 = self.encoder_block_04_2(out_enc4_1)
        # out_enc4_3 = self.encoder_block_04_3(out_enc4_2)


        # print("out_enc4 ::::  ", out_enc4.shape)
        feature_prop_3 = self.feature_propagator_3(out_upper_down6, out_enc4)

        # cat3 = torch.cat([out_enc4, out_upper_down6], dim=1)
        # print("feature_prop_3 ::::  ", feature_prop_3.shape)
        up1      = self.enc_ups1(feature_prop_3)
        # print("up1 ::::  ", up1.shape)
        out_enc5 = self.encoder_block_05(up1)
        # out_enc5_1 = self.encoder_block_05_1(out_enc5)
        # out_enc5_2 = self.encoder_block_05_2(out_enc5_1)
        # print("out_enc5 ::::  ", out_enc5.shape)
        feature_prop_4 = self.feature_propagator_4(out_upper_down8, out_enc5)
        # cat4 = torch.cat([out_enc5, out_upper_down8], dim=1)
        # print("feature_prop_4 ::::  ", feature_prop_4.shape)
        up2      = self.enc_ups2(feature_prop_4)
        # print("up2 ::::  ", up2.shape)
        out_enc6 = self.encoder_block_06(up2)
        # out_enc6_1 = self.encoder_block_06_1(out_enc6)


        # print("out_enc6 ::::  ", out_enc6.shape)
        feature_prop_5 = self.feature_propagator_5(out_upper_down9, out_enc6)
        # cat5 = torch.cat([out_enc6, out_upper_down9], dim=1)
        # print("feature_prop_5 ::::  ", feature_prop_5.shape)
        up3      = self.enc_ups3(feature_prop_5)
        # print("up3 ::::  ", up3.shape)
        out_enc7 = self.encoder_block_07(up3, DAB5)
        # print("out_enc7 ::::  ", out_enc7.shape)
        cat6 = torch.cat([out_enc7, out_upper6], dim=1)
        # print("cat6 ::::  ", cat6.shape)
        

        # visualize_featuremaps_batchwise(cat6, "cat6.png")
        # exit(0)

        # out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        # out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        # out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        # out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        
        # fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        # fr = self.refinement(fd)
        out = self.output(cat6) + x
        return out, DAB5
