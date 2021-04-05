import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
# from data_pytorch import Data

from resnet import ResNet
from data import CIFAR

import time
import shutil
import yaml
import argparse

parser = argparse.ArgumentParser(
    description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def train(train_loader, model, criterion, optimizer, epoch):
	total_loss = 0
	for i, (input, target) in enumerate(train_loader):
		#print(input.shape)
		#print(target.shape)
		optimizer.zero_grad()

		predicted_label = model.forward(input)

		loss = criterion(predicted_label, target)
		loss.backward()
		optimizer.step()
		total_loss += loss
	return total_loss


def validate(val_loader, model, criterion):
	total_loss = 0
	for i, (input, target) in enumerate(val_loader):
		predicted_label = model.forward(input)
		loss = criterion(predicted_label, target)
		total_loss += loss
	return total_loss

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
	torch.save(state, filename)
	# best_one stores whether your current checkpoint is better than the previous checkpoint
	if best_one:
		shutil.copyfile(filename, filename2)

def main():
	n_epochs = config["num_epochs"]
	model = ResNet(0,0,0) #make the model with your paramters
    
	criterion = nn.CrossEntropyLoss() #what is your loss function
    
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #which optimizer are you using

	train_dataset = './images/' + '*'#how will you get your dataset
	train_loader = CIFAR(train_dataset) # how will you use pytorch's function to build a dataloader

	val_dataset = './validation/' + '*' #how will you get your dataset
	val_loader = CIFAR(val_dataset) # how will you use pytorch's function to build a dataloader

	current_best_validation_loss = float("Infinity")

	print(len(train_loader))

	for epoch in range(n_epochs):
		total_loss = train(train_loader, model, criterion, optimizer, epoch)
		print("Epoch {0}: {1}".format(epoch, total_loss))
		validation_loss = validate(val_loader, model, criterion)
		print("\tTest Accuracy {0}".format(validation_loss))
		if validation_loss < current_best_validation_loss:
			save_checkpoint(model.state_dict(), True)
			current_best_validation_loss = validation_loss
		else:
			save_checkpoint(model.state_dict(), False)


if __name__ == "__main__":
    main()
