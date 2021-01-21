#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from util_hourglass import *
from torch.quantization import QuantStub, DeQuantStub

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()
        self.resizing = resize_layer(3, 32)
        # feature extraction
        self.layer1 = hourglass_block(32, 32)
        self.layer2 = hourglass_block(32, 32)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def fuse_model(self):
        modules_list = [m for m in self.modules()]
        for i, m in enumerate(modules_list):
            if type(m) == Conv2D_BatchNorm_Relu:
                if type(m.cbr_unit) == nn.Sequential and len(m.cbr_unit) == 3:
                    for idx in range(len(m.cbr_unit)):
                        if type(m.cbr_unit[idx]) == nn.Conv2d:
                            torch.quantization.fuse_modules(m.cbr_unit, [str(idx), str(idx + 1), str(idx + 2)],
                                                            inplace=True)
            if type(m) == UNet_2Plus:
                m.fuse_model()
            if type(m) == nn.Sequential and len(m) == 2:
                torch.quantization.fuse_modules(m, ["0", "1"], inplace=True)

    def forward(self, inputs):
        # feature extraction
        inputs = self.quant(inputs)
        out = self.resizing(inputs)
        result1, out, feature1 = self.layer1(out)
        result2, out, feature2 = self.layer2(out)
        for i,res in enumerate(result1):
            result1[i] = self.dequant(res)
        for i,res in enumerate(result2):
            result2[i] = self.dequant(res)
        feature1 = self.dequant(feature1)
        feature2 = self.dequant(feature2)
        return (result1, result2), (feature1, feature2)