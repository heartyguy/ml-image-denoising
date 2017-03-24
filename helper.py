import math

from PIL import Image
import numpy
import scipy


def cmp_images_psnr(truth_array, noisy_array, length):
    psnr_val = 0.0
    for i in range(length):
        psnr_val += psnr(truth_array[i], noisy_array[i], 255.0)
    return psnr_val/length



def psnr(dataset1, dataset2, maximumDataValue, ignore=None):
   # Make sure that the provided data sets are numpy ndarrays, if not
   if type(dataset1).__module__ != numpy.__name__:
      d1 = numpy.asarray(dataset1).flatten()
   else:
      d1 = dataset1.flatten()

   if type(dataset2).__module__ != numpy.__name__:
      d2 = numpy.asarray(dataset2).flatten()
   else:
      d2 = dataset2.flatten()

   # Make sure that the provided data sets are the same size
   if d1.size != d2.size:
      raise ValueError('Provided datasets must have the same size/shape')

   # Check if the provided data sets are identical, and if so, return an
   # infinite peak-signal-to-noise ratio
   if numpy.array_equal(d1, d2):
      return float('inf')

   # If specified, remove the values to ignore from the analysis and compute
   # the element-wise difference between the data sets
   if ignore is not None:
      index = numpy.intersect1d(numpy.where(d1 != ignore)[0], 
                                numpy.where(d2 != ignore)[0])
      error = d1[index].astype(numpy.float64) - d2[index].astype(numpy.float64)
   else:
      error = d1.astype(numpy.float64)-d2.astype(numpy.float64)

   # Compute the mean-squared error
   meanSquaredError = numpy.sum(error**2) / error.size

   # Return the peak-signal-to-noise ratio
   return 10.0 * numpy.log10(maximumDataValue**2 / meanSquaredError)


def bicubic_interpolate(image_np):
    image_arr = numpy.asarray(image_np)
    red = image_arr[:][:][0]
    green = image_arr[:][:][1]
    blue = image_arr[:][:][2]
    scipy.interpolate.interp2d()
   