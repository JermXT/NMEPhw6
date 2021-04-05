
import torch.nn as nn
#add imports as necessary

class ResNet:

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
        self.avgpool = nn.AvgPool2d((7,7))
        self.fc = nn.Linear(512, 1000)


    def forward(self, x):
        #TODO: implement the forward function for resnet,
        #use all the functions you've made
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # may need flattening?
        x = self.fc(x)
        return x

    def new_block(self, in_planes, out_planes, stride):
        layers = [nn.Conv2d(in_planes, out_planes, (3,3), stride), nn.BatchNorm2d(out_planes), nn.ReLu(), nn.Conv2d(out_planes, out_planes, (3,3), stride)), nn.BatchNorm2d(planes), nn.ReLU()]
        #TODO: make a convolution with the above params
        return nn.Sequential(*layers)
