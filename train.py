import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import get_celeba, get_mnist, show_tensor_images
from dcgan import weights_init, Generator, Discriminator 
from torchvision.utils import save_image



# Parameters to define the model.
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str , default='mnist', required=True, help='mnist | celebA')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nc', type=int, default=1, help='Number of channles in the training images. For coloured images this is 3.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nepochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=369, help='manual seed')
parser.add_argument('--save_epoch', type=int, default=2, help='save data after number of epoch')


opt = parser.parse_args()
print(opt)
os.makedirs("images/%s" % opt.dataset, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset, exist_ok=True)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

if opt.dataset == 'celebA' :
    dataloader = get_celeba(opt)
else :
    dataloader = get_mnist(opt)

# # Plot the training images.
# sample_batch = next(iter(dataloader))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(
#     sample_batch[0].to(device)[ : 64], padding=2, normalize=True).cpu(), (1, 2, 0)))

# plt.show()

# Create the generator.
gen = Generator(opt).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
gen.apply(weights_init)
# Print the model.
print(gen)

# Create the discriminator.
disc = Discriminator(opt).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
disc.apply(weights_init)
# Print the model.
print(disc)

# Binary Cross Entropy loss function.
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizer for the discriminator.
optimizerD = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# Optimizer for the generator.
optimizerG = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

iters = 0

print("Starting Training Loop...")
print("-"*25)

for epoch in range(opt.nepochs):
    loop = tqdm(enumerate(dataloader, 0), total= len(dataloader))
    for i, data in loop:
        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from opt.batchSize for last batch in epoch.
        b_size = real_data.size(0)
        
        # Make accumalated gradients of the discriminator zero.
        disc.zero_grad()
        output = disc(real_data).reshape(-1)
        errD_real = criterion(output, torch.ones_like(output))
        # Calculate gradients for backpropagation.
        errD_real.backward()
        D_x = output.mean().item()
        
        # Sample random data from a unit normal distribution.
        noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
        # Generate fake data (images).
        fake_data = gen(noise)
        # Calculate the output of the discriminator of the fake data.
        # As no gradients w.r.t. the generator parameters are to be
        # calculated, detach() is used. Hence, only gradients w.r.t. the
        # discriminator parameters will be calculated.
        # This is done because the loss functions for the discriminator
        # and the generator are slightly different.
        output = disc(fake_data.detach()).reshape(-1)
        errD_fake = criterion(output, torch.zeros_like(output))
        # Calculate gradients for backpropagation.
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Net discriminator loss.
        errD = errD_real + errD_fake
        # Update discriminator parameters.
        optimizerD.step()
        
        # Make accumalted gradients of the generator zero.
        gen.zero_grad()
        # We want the fake data to be classified as real. Hence
        # real_label are used. (label=1)
        # No detach() is used here as we want to calculate the gradients w.r.t.
        # the generator this time.
        output = disc(fake_data).view(-1)
        errG = criterion(output, torch.ones_like(output))
        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by disc.zero_grad()
        errG.backward()

        D_G_z2 = output.mean().item()
        # Update generator parameters.
        optimizerG.step()

        # Check progress of training.
        loop.set_description(f"Epoch [{epoch + 1}/{opt.nepochs}]")
        loop.set_postfix({
            "loss_D" :"{:.4f} ".format(errD.item()),
            "loss_G":"{:.4f} ".format(errG.item()),
            "D(x)":  D_x,
            "D(G(z))": f"{D_G_z1 / D_G_z2}"
        })

        # Save the losses for plotting.
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == opt.nepochs) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = gen(fixed_noise)
                save_image(fake_data, "images/%s/%s.png" % (opt.dataset, iters), normalize=True)
            img_list.append(vutils.make_grid(fake_data.detach().cpu(), padding=2, normalize=True))

        iters += 1

    # Save the model.
    if epoch % opt.save_epoch == 0:
        torch.save({
            'generator' : gen.state_dict(),
            'discriminator' : disc.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : opt
        }, "saved_models/%s/model_epoch_%s.pth" % (opt.dataset, epoch))

# Save the final trained model.
torch.save({
            'generator' : gen.state_dict(),
            'discriminator' : disc.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : opt
}, 'saved_models/{}/model_final.pth' % (opt.dataset)
)

# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# plt.show()
# anim.save('celeba.gif', dpi=80, writer='imagemagick')
writer = animation.FFMpegWriter(
    fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim.save("movie.mp4", writer=writer)

plt.show()
