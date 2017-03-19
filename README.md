# ml-image-denoising
Image denoising using PCA, NMF, SVD, Spectral decomposition, CNN and state of the art generative adversarial denoising autoencoder

## How to run
require pip, virtualenv and git

```./install.sh```

You should download the CelebA dataset from [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (you're looking for a file called img_align_celeba.zip).  Unzip into the parent directory /.. then run

``` python dataprocessing.py ```

This will crop the images to the right size and store them in HDF5 format.

Currently, we only process 5 batches * 1024 images

Next run the dcgan notbook.

``` jupyter notebook ```
