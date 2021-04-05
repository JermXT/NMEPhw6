import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
# from data_pytorch import Data

from resnet import ResNet
from data import CIFAR

import numpy as np

import time
import shutil
import yaml
import argparse

from torchsummary import summary



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
	model.train()
	total_loss = 0
	#print("hi")
	for i, (input, target) in enumerate(train_loader):
		#print(input.shape)
		#print(target.shape)
		optimizer.zero_grad()
		#print(input.shape)
		input = input.cuda()
		predicted_label = model.forward(input).cuda()
		target = target.cuda()
		loss = criterion(predicted_label, target).cuda()
		loss.backward()
		optimizer.step()
		total_loss += loss
	return total_loss


def validate(val_loader, model, criterion):
	model.eval()
	with torch.no_grad():
	    total_loss = 0
	    total = 0
	    accuracies = 0
	    for i, (input, target) in enumerate(val_loader):
		    #print(input.shape)
		    #print(target.shape)
		    input = input.cuda()
		    target = target.cuda()
		    predicted_label = model.forward(input).cuda()
		    loss = criterion(predicted_label, target).cuda()
		    total += 128
		    accuracies += torch.sum(target == torch.argmax(predicted_label, dim=1))
		    total_loss += loss
	    return total_loss, accuracies/total

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
	torch.save(state, filename)
	# best_one stores whether your current checkpoint is better than the previous checkpoint
	if best_one:
		shutil.copyfile(filename, filename2)

def main():
	print(torch.cuda.device_count(), "gpus available")
	
	# print(summary(model, (3, 32, 32)))
	print("HEYO")

	n_epochs = config["num_epochs"]
	print(n_epochs)
	model = ResNet(0,0,0).cuda() #make the model with your paramters
    
	criterion = nn.CrossEntropyLoss() #what is your loss function
    
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #which optimizer are you using

	train_dataset = './images/' + '*'#how will you get your dataset
	train_loader = CIFAR(train_dataset) # how will you use pytorch's function to build a dataloader
	
	# next --> (4x3x32x32, 4)
	"""
	image_stack = []
	label_stack = []
	for i in range(32):
		temp = next(iter(data_loader))
		image_stack.append(temp[0])
		label_stack.append(temp[1])
	img = np.concatenate(image_stack, axis=0)
	lb = np.concatenate(label_stack, axis=0)
	batch_1 (img, lb)
	"""
	print("hi")
	all_batches = []
	d_iter = iter(train_loader)
	for i in range(45000//32):
		image_stack = []
		label_stack = []
		for i in range(32):
			temp = next(d_iter)
			image_stack.append(temp[0])
			label_stack.append(temp[1])
		img = torch.cat(image_stack, axis=0)
		lb = torch.cat(label_stack, axis=0)
		all_batches.append((img, lb))

	print("concat done")


	val_dataset = './validation/' + '*' #how will you get your dataset
	val_loader = CIFAR(val_dataset) # how will you use pytorch's function to build a dataloader
	
	val_batches = []
	d_iter = iter(val_loader)
	for i in range(4900//32):
		image_stack = []
		label_stack = []
		for i in range(32):
			temp = next(d_iter)
			image_stack.append(temp[0])
			label_stack.append(temp[1])
		img = torch.cat(image_stack, axis=0)
		lb = torch.cat(label_stack, axis=0)
		val_batches.append((img, lb))

	current_best_validation_loss = float("Infinity")

	print(len(train_loader))

	for epoch in range(n_epochs):
		total_loss = train(all_batches, model, criterion, optimizer, epoch)
		print("Epoch {0}: {1}".format(epoch, total_loss))
		validation_loss, accuracy = validate(val_batches, model, criterion)
		print("Test Loss {0}".format(validation_loss))
		print("Test Accuracy {0}".format(accuracy))
		if accuracy < current_best_validation_loss:
			save_checkpoint(model.state_dict(), True)
			current_best_validation_loss = accuracy
		else:
			save_checkpoint(model.state_dict(), False)


if __name__ == "__main__":
    main()
