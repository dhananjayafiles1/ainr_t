import torch.nn as nn
import torch


class model(nn.Module):
	def __init__(self, kernel_size, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv = nn.Conv2d(100, 500, kernel_size=kernel_size)
	
	def forward(self, x):
		y = self.conv(x)
		return y

x = torch.randn(50, 100, 200, 300)
device = torch.device("cuda")
network = model(3)
network.to(device)
network = nn.DataParallel(network, device_ids=[4, 5])

while(True):
	y = network(x)
