import torch
import lpips as lpips_lib

class LPIPS(torch.nn.Module):
    def __init__(self, net='vgg'):
        super(LPIPS, self).__init__()
        self.metric = lpips_lib.LPIPS(net=net)

    def forward(self, x, y):
        return self.metric(x, y)
