
import torch.nn as nn

#add imports as necessary

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=3)
        #i think there is padding^?
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU() 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.new_block(64,64,1)
        self.layer2 = self.new_block(64,128,2)
        self.layer3 = self.new_block(128,256,2)
        self.layer4 = self.new_block(256,512,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 4)
        # self.softMax = nn.Softmax2d()

    def forward(self, x):
        #TODO: implement the forward function for resnet,
        #use all the functions you've made
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("After relu:" + str(x.shape))
        x = self.maxpool(x)
        # print("After maxpool:" + str(x.shape))
        x = self.layer1(x)
        # print("Layer 1 output:" + str(x.shape))
        x = self.layer2(x)
        # print("Layer 2 output:" + str(x.shape))
        x = self.layer3(x)
        # print("Layer 3 output:" + str(x.shape))
        x = self.layer4(x)
        # print("Layer 4 output:" + str(x.shape))
        x = self.avgpool(x)
        # print("After Average Pooling:" + str(x.shape))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print("After FC" + str(x.shape))
        # x = self.softMax(x)
        return x

    def new_block(self, in_planes, out_planes, stride):
        layers = [nn.Conv2d(in_planes, out_planes, (3,3), stride, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(), nn.Conv2d(out_planes, out_planes, (3,3), stride=1, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU()]
        #TODO: make a convolution with the above params
        return nn.Sequential(*layers)
