import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, fftshift, ifft
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import misc
from PIL import Image
from scipy.interpolate import interp1d
import argparse
from functools import partial


# Apply circular mask to image
def maskImage(image, radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x ** 2 + y ** 2 > radius ** 2
    image[mask] = 0
    return image

def projection(img, thetas):
  '''Forward projection '''
  img = Image.fromarray(img)
  num_theta = len(thetas)
  sinogram = np.zeros((img.size[1], num_theta))
  for i, theta in enumerate(thetas):
    rot_img = img.rotate(theta)
    sinogram[:, i] = np.sum(rot_img, axis=0)
  return sinogram

def get_ramp_fourier_filter(size):
  '''Ramp filter in fourier space'''
  n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                      np.arange(size / 2 - 1, 0, -2, dtype=int)))
  f = np.zeros(size)
  f[0] = 1 / 4
  f[1::2] = -1 / (np.pi * n) ** 2
  fourier_filter = np.real(fft(f)) 
  return fourier_filter[:, np.newaxis]

def filtered_projection(sinogram):
  ''' Sinogram filtration with ramp filter'''
  size, thetas = sinogram.shape
  # resize image to next power of two (but no less than 64) for
  # Fourier analysis; speeds up Fourier and lessens artifacts
  projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * size))))
  pad_width = ((0, projection_size_padded - size), (0, 0))
  sinogram = np.pad(sinogram, pad_width, mode='constant', constant_values=0)
  fft_ramp = get_ramp_fourier_filter(sinogram.shape[0])
  projection = fft(sinogram, axis=0) * fft_ramp
  filtered = np.real(ifft(projection, axis=0)[:size, :])
  return filtered

def cuttering(sinogram, rate=0.5):
  '''Cut chosen rate of threads '''
  cutting_sinogram = sinogram.copy()
  len_size = sinogram.shape[0]
  idx = np.random.choice(len_size, size=int(rate*len_size), replace=False)
  cutting_sinogram.T[:, idx] = 0
  return cutting_sinogram

def back_projection(filtered, theta):
  output_size, len_theta = filtered.shape
  reconstructed = np.zeros((output_size, output_size))
  radius = output_size // 2
  xpr, ypr = np.mgrid[:output_size, :output_size] - radius
  x = np.arange(output_size) - output_size // 2

  for col, angle in zip(filtered.T, np.deg2rad(theta)):
      t = ypr * np.cos(angle) + xpr * np.sin(angle)
      interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
      reconstructed += interpolant(t)
  out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
  reconstructed[out_reconstruction_circle] = 0.
  return reconstructed

def binarize(sinogram):
  ''' Random dithering 
  For each pixel set 1 with probability in this pixel, else 0 
  '''
  hm, wm = sinogram.shape
  for h in range(hm):
    for w in range(wm):
      p = sinogram[h, w]
      sinogram[h, w] = np.random.choice([0, 1], p = [1 - p, p])
  return sinogram

def normalize(sinogram):
  ''' Normalization '''
  sinogram[sinogram < 0] = 0
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler = scaler.fit_transform(sinogram)
  scaler = np.where(scaler > 1, 1, scaler)
  return scaler

def contrast(img, compare_param = 0.6):
  ''' Contrast pixels with high brightness '''
  return np.where(img > np.mean(img)*compare_param, img*10000, img)

def preprocess_image(img_path, imgRadius = 200):
  '''Convert to gray, crop, resize and create circle mask '''
  # Load image
  image = cv2.imread(img_path)
  # Crop image
  height, width = image.shape[0:2]
  minEdge= min(height, width)
  topEdge = int((height - minEdge)/2)
  leftEdge = int((width - minEdge)/2)
  imgCropped = image[100:1000, 150:1050]
  # Convert to grayscale
  imgGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
  # Resize image
  imgSized = cv2.resize(imgGray, (2*imgRadius + 1, 2*imgRadius + 1))
  # Mask image
  imgMasked = maskImage(imgSized, imgRadius)
  return imgMasked[:401, :401]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--path',
      default='photo.jpg',
      help='Path to file'
  )
  parser.add_argument(
      '--number_angles',
      default=60,
      help='Number of angles'
  )
  parser.add_argument(
      '--rate',
      default=0.5,
      help='Rate of threads to delete'
  )
  parser.add_argument(
      '--visualize',
      default=True,
      help='Rate of threads to delete'
  )
  parser.add_argument(
      '--contrast',
      default=True,
      help='Rate of threads to delete'
  )
  args = parser.parse_args()
  num_angles = int(args.number_angles)
  rate = float(args.rate)
  is_contrast = args.contrast
  visualize = args.visualize
  img_path = args.path
  print('Start image preprocessing')
  img = preprocess_image(img_path)
  if visualize:
    plt.title('Initial image')
    plt.imshow(img, cmap='gray')
    plt.show()
  theta = np.linspace(0., 360., num_angles)
  print('Compute the sinogram')
  sinogram = projection(img, theta)
  if visualize:
    plt.title('Initial sinogram')
    plt.imshow(sinogram.T, cmap='gray')
    plt.show()
  print('Filter the sinogram with ramp filter')
  filtered = filtered_projection(sinogram)
  if visualize:
    plt.title('Filtered sinogram')
    plt.imshow(filtered.T, cmap='gray')
    plt.show()
  print('Cut defined rate of threads from the sinogram')
  cuttered = cuttering(filtered, rate)
  if visualize:
    plt.title('Cuttered sinogram')
    plt.imshow(cuttered.T, cmap='gray')
    plt.show()
  print('Normalize and binarize filtered sinogram')
  normalized = normalize(cuttered)
  if visualize:
    plt.title('Normalized sinogram')
    plt.imshow(normalized.T, cmap='gray')
    plt.show()
  binarized = binarize(normalized)
  if visualize:
    plt.title('Binarized sinogram')
    plt.imshow(binarized.T, cmap='gray')
    plt.show()
  print('Backprojection of obtained sinogram')
  backprojected = back_projection(binarized, theta)
  if visualize:
    plt.title('After back projection')
    plt.imshow(backprojected, cmap='gray')
    plt.show()
  cv2.imwrite("result.png", (backprojected-backprojected.min())*255/(backprojected.max()-backprojected.min()))
  if is_contrast:
    print('Contrasting obtained backprojection')
    contrasted = contrast(backprojected)
    cv2.imwrite("contrasted_result.png", (contrasted-contrasted.min())*255/(contrasted.max()-contrasted.min()))
    if visualize:
        plt.title('After contrasting')
        plt.imshow(contrasted, cmap='gray')
        plt.show()

main()