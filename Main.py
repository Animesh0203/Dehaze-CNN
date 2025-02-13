import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models

# Define the SSIM loss function
def ssim_loss(output, target):
    return 1 - F.ssim(output, target)

# Define the Perceptual loss class
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18])  # Adjust as needed

    def forward(self, x, y):
        features_x = self.feature_extractor(x)
        features_y = self.feature_extractor(y)
        return F.mse_loss(features_x, features_y)

if __name__ == '__main__':
    from Dataset import HazeDataset

# Define your model architecture
    class DehazeNet(nn.Module):
        def __init__(self):
            super(DehazeNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=3)
            self.bn4 = nn.BatchNorm2d(512)
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.b = 1

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            k = F.relu(self.conv5(x))

            if k.size() != x.size():
                raise Exception("k, haze image are different size!")

            output = k * x - k + self.b
            return F.relu(output)
        

    # Set your hyperparameters
    batch_size = 16
    learning_rate = 1e-3
    scheduler_step_size = 10
    scheduler_gamma = 0.5
    num_epochs = 50

    # Initialize your model, loss, and optimizer
    device = torch.device('cuda')
    model = DehazeNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Create data loaders for training and validation
    transform = transforms.Compose([
        transforms.Resize([480, 640]),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    train_dataset = HazeDataset("D:\Projects\Machine Learning\Dehaze(We modding it again)\Revside Dataset\Train", transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Set up TensorBoard
    writer = SummaryWriter()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for step, (ground_truth_img, hazy_img) in enumerate(train_loader):
            ground_truth_img, hazy_img = ground_truth_img.to(device), hazy_img.to(device)

            # Forward pass
            dehaze_img = model(hazy_img)
            # loss = criterion(dehaze_img, ground_truth_img)

            # Compute the loss using SSIM loss
            loss = ssim_loss(dehaze_img, ground_truth_img)
        
            # Compute the loss using Perceptual loss
            # loss = perceptual_criterion(dehaze_img, ground_truth_img)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item()}')
                print(f'lr = {learning_rate}')

                # Log the loss to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + step)

                # Log hazy images, gt, and model outputs as images to TensorBoard
                if step % 500 == 0:
                    # Create a grid of images
                    grid_images = torchvision.utils.make_grid(
                        torch.cat([hazy_img, ground_truth_img, dehaze_img], dim=0), nrow=batch_size
                    )
                    writer.add_image('Images/hazy_gt_dehaze', grid_images, epoch * len(train_loader) + step)

    # Update the learning rate
    scheduler.step()

    # Close the TensorBoard writer
    writer.close()

    # Save the trained model
    torch.save(model.state_dict(), 'ModelsFinished\DehazeNet_Model.pth')
