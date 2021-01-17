import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

def get_mnist(opt):
    """
    Loads the MNIST dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(opt.nc)], [0.5 for _ in range(opt.nc)]),
    ])
    
    dataset = datasets.MNIST(root=f"dataset/mnist", train=True, transform=transform,
                       download=True)
    # Create the dataloader.
    dataloader = DataLoader(dataset,
        batch_size= opt.batchSize,
        shuffle=True)
    
    return dataloader
    

def get_celeba(opt):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Directory containing the data.
    # os.system('cmd /k "color a & date"')

    root = 'dataset/data_faces/'
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])
    
    # Create the dataset.
    dataset = datasets.ImageFolder(root=root, transform=transform)
    
    # Create the dataloader.
    dataloader = DataLoader(dataset,
        batch_size=opt.batchSize,
        shuffle=True
    )
    
    return dataloader

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()  
