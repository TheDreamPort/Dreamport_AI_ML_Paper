import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import build_mlp
from load import build_dataloaders
import files as f
import pickle
import math
import time
import copy
import sys
import argparse
import progressbar
import torch


def train(model, attrib_dict, dataloaders, args):
    '''Trains a neural network model on the training data, evaluating its performance
    on the test data after each epoch, and returning the most performant model.
    Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html.
    '''
    loss_function = nn.NLLLoss()
    log_soft_max = nn.LogSoftmax(dim=1)
    optimizer = optim.Adam(model.parameters(), lr=args.learningrate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs / 4, args.epochs / 2, 3 * args.epochs / 4],
        gamma=0.5
    )
    batch_size = model.dataloader.batch_size
    output_size = model.output_size
    model.train()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_neg = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            negs = 0
            true_negs = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # Convert to gpu if necessary
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Track history only if in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model.forward(inputs)
                    _, preds = torch.max(outputs, 1)
                    outputs = log_soft_max(outputs)
                    loss = loss_function(outputs, labels)
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Generate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                true_negs += ((preds == 0) + (preds == labels.data)).eq(2).sum()
                negs += torch.sum(labels.data == 0)

            epoch_loss = running_loss / attrib_dict[phase].num_examples
            epoch_acc = running_corrects.double() / attrib_dict[phase].num_examples

            epoch_true_neg = true_negs.double() / negs

            print('{} Loss: {:.4f} Acc: {:.4f} TN: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_true_neg))

            if phase == 'train':
                scheduler.step()
                print('Learning rate: {:.6f}'.format(get_lr(optimizer)))
            else:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if not args.maximizetrueneg:
                        best_model_wts = copy.deepcopy(model.state_dict())

                if epoch_true_neg > best_neg:
                    best_neg = epoch_true_neg
                    if args.maximizetrueneg:
                        best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val True Negative: {:4f}'.format(best_neg))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save(model, save_file):
    torch.save(model.state_dict(), save_file)


def main():
    '''Loads data, trains a neural network on the data, and saves the model.'''

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Trains a multilayer perceptron model on the NSL-KDD dataset.")
    parser.add_argument("-lr", "--learningrate", type=float,
                        help="Initial learning rate of neural network.", default=0.001)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of passes over the network during training.", default=700)
    parser.add_argument("-tn", "--maximizetrueneg", type=bool,
                        help="Choose model with best true negative rate rather than best accuracy.",
                        default=False)
    args = parser.parse_args()

    # Load data, train model, and save model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on " + str(device) + ".")
    dataloaders, attrib_dict = build_dataloaders()
    save(train(build_mlp(dataloaders, attrib_dict, device), attrib_dict,
               dataloaders, args), f.MLP)


if __name__ == '__main__':
    main()
