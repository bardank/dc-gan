# Deep Convolutional GAN

PyTorch implementation of DCGAN introduced in the paper: [Unsupervised Representation Learning with Deep Convolutional 
Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), Alec Radford, Luke Metz, Soumith Chintala.

<p align="center">
<img src="result.gif" title="Generated Data Animation" alt="Generated Data Animation">
</p>


## Data
This implementation uses the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. However, any other dataset can
also be used. Download the data and update the directory location inside the `root` variable in **`utils.py`**. Also don't forget to change the number of channels in **`train.py`** according to the channels of images in your dataset.

## Downloading CelebA dataset
You can download the CelebA dataset by using the following code in terminal:
```
mkdir data_faces`<br/>
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip 
```
## Extracting CelebA dataset into your datset folder
```
import zipfile
with zipfile.ZipFile("celeba.zip","r") as zip_ref:
   zip_ref.extractall("data_faces/")
```
## References
1. **Alec Radford, Luke Metz, Soumith Chintala.** *Unsupervised representation learning with deep convolutional 
generative adversarial networks.*[[arxiv](https://arxiv.org/abs/1511.06434)]
2. **Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
Sherjil Ozair, Aaron Courville, Yoshua Bengio.** *Generative adversarial nets.* NIPS 2014 [[arxiv](https://arxiv.org/abs/1406.2661)]
3. **Ian Goodfellow.** *Tutorial: Generative Adversarial Networks.* NIPS 2016 [[arxiv](https://arxiv.org/abs/1701.00160)]
4. DCGAN Tutorial. [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html]
5. PyTorch Docs. [https://pytorch.org/docs/stable/index.html]
6. Natsu6767. [https://github.com/Natsu6767/DCGAN-PyTorch]
