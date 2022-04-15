import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

# Option 1
class MyNet(nn.Module): 
    def __init__(self, features_p1, features_p2, num_output=2500, init_weights=False):
        super(MyNet, self).__init__()
        self.features_p1 = features_p1
        self.features_p2 = features_p2
        self.regressor = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(512 * 5 * 1, num_output), # 1024 -> 2500
            # nn.BatchNorm1d(2048),
            # nn.LeakyReLU(inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Input: (N, 1, 2, 8000)
        x = self.features_p1(x)
        # (N, 512, 2, 125)
        x = self.features_p2(x)
        # (N, 512, 2, 1)
        x = torch.flatten(x, start_dim=1) # torch.flatten() or out.view()
        # x = x.view(-1, 512*5)
        # (N, 512 * 2 * 1 = 1024)
        x = self.regressor(x)
        # x = F.softmax(x, dim=1)
        # x = F.sigmoid(x)
        # Output: (N, 2500)
        return x


def make_features_p1(cfg: list):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=(1, 2), stride=(1,2))] # H = H, W = floor (W / 2)
            # layers += [nn.AvgPool2d(kernel_size=(1, 2), stride=(1,2))] # H = H, W = floor (W / 2)
        # elif v == 8:
        #     conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1) # H = H, W = W 
        #     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        #     in_channels = v
        elif v == 16:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1) # H = H, W = W 
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] 
            in_channels = v
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 3), stride=(1,1), padding=(0, 1)) # H = H, W = W 
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_features_p2(cfg: list):
    layers = []
    in_channels = 512
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=(1, 2), stride=(1,2))] # H = H, W = floor (W / 2)
            # layers += [nn.AvgPool2d(kernel_size=(1, 2), stride=(1,2))] # H = H, W = floor (W / 2)
        elif v == 256:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1) # H = H, W = W 
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 512:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 3), stride=(1,1), padding=(0, 1)) # H = H, W = W 
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            # layers += [conv2d, nn.BatchNorm2d(v), nn.ELU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    
    # 'vgg_p1': [8, 16, 'M', 32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M' ],
    'vgg_p1': [16, 32, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M' ],
    # 'vgg_p1': [16, 32, 32, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M' ],
    'vgg_p2': [256, 512, 'M', 256, 512, 'M', 256, 512, 'M', 256, 512, 'M', 256, 512, 'M', 256, 512, 'M'],
    # 'vgg_p2': [256, 512, 256, 512, 'M', 256, 512, 256, 512, 'M', 256, 512, 256, 512, 'M', 256, 512, 256, 512, 'M', 256, 512, 256, 512, 'M', 256, 512, 256, 512, 'M'],
     
}



def mynet(model_name_p1="vgg_p1", model_name_p2="vgg_p2", **kwargs):
    assert model_name_p1 in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name_p1)
    assert model_name_p2 in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name_p2)
    cfg_p1 = cfgs[model_name_p1]
    cfg_p2 = cfgs[model_name_p2]
    model = MyNet(make_features_p1(cfg_p1),make_features_p2(cfg_p2), **kwargs)
    return model


net = mynet()

CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()

summary(net, input_size=(1, 5, 8000))

