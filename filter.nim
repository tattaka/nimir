import math, strutils, sequtils, random, typetraits, future, macros, os

import arraymancer
var
  img = read_image("images/lena.png") # Read image

proc add_noise*(img: Tensor[uint8]): Tensor[uint8]=
  var
    noise: seq[float]
    noise_img: Tensor[float]
  result = newTensor[uint8]([img.shape[0], img.shape[1], img.shape[2]])
  randomize()
  for i in 0..<img.shape[0]*img.shape[1]*img.shape[2]:
    noise.add(rand(1.0))
  noise_img = noise.toTensor().reshape(img.shape[0], img.shape[1], img.shape[2])
  for c in 0..<img.shape[0]:
    for h in 0..<img.shape[1]:
      for w in 0..<img.shape[2]:
        result[c, h, w] = uint8(noise_img[c, h, w] * float(img[c, h, w]))
  return result
var noise_img = add_noise(img)
write_png(noise_img, "images/lena_noise.png")

proc correlation*(img: Tensor[uint8], kernel: Tensor[float], stride = 1, padding = 0): Tensor[uint8] =
  assert(kernel.shape.len == 2)
  assert(kernel.shape[0] mod 2 == 1)
  assert(kernel.shape[1] mod 2 == 1)
  assert(padding >= 0)
  let
    stride = stride
    padding = padding
    i_w = img.shape[2]
    i_h = img.shape[1]
    k_w = kernel.shape[1]
    k_h = kernel.shape[0]
    output_w = (i_w + 2*padding - k_w) div stride + 1
    output_h = (i_h + 2*padding - k_h) div stride + 1
  var
    img = img
    pad_img = zeros[uint8]([img.shape[0], i_h+2*padding, i_w+2*padding])


  result = newTensor[uint8]([img.shape[0], output_h, output_w])
  for c in 0..<img.shape[0]:
    for i in 0..<i_h:
      for j in 0..<i_w:
        pad_img[c, i+padding, j+padding] = img[c, i, j]

  img = pad_img
  for c in 0..<result.shape[0]:
    for i in 0..<result.shape[1]:
      for j in 0..<result.shape[2]:
        var I_xy = 0.0
        for x in 0..<k_h:
          for y in 0..<k_w:
            I_xy = I_xy + kernel[x, y] * float(img[c, i*stride+x, j*stride+y])
        result[c, i, j] = uint8(I_xy)


proc smoothing*(img: Tensor[uint8], ksize = 3, stride = 1, padding = 0): Tensor[uint8] =
  var
    kernel = ones[float]([ksize, ksize]) / 9
  echo kernel
  result = correlation(img, kernel, stride, padding)

write_png(smoothing(img), "images/lena_smooth.png")

proc gaussian*(img: Tensor[uint8], ksize = 3, stride = 1, padding = 0, scale = 1.0): Tensor[uint8] =
  var
    kernel = newTensor[float](ksize, ksize)
  for i in 0..<ksize:
    for j in 0..<ksize:
      var
        x = i - ksize div 2
        y = j - ksize div 2
      kernel[i, j] = exp(-float(x^2+y^2)/(2*(scale^2))) / (2*PI*(scale^2))
  kernel = kernel / kernel.sum
  echo kernel
  result = correlation(img, kernel, stride, padding)

write_png(gaussian(noise_img, ksize = 11, scale = 3.0), "images/lena_gaussian.png")

proc bilateral*(img: Tensor[uint8], ksize = 3, stride = 1, padding = 0, scale_d = 1.0, scale_r = 1.0): Tensor[uint8] =
  assert(ksize mod 2 == 1)
  assert(padding >= 0)
  let
    stride = stride
    padding = padding
    i_w = img.shape[2]
    i_h = img.shape[1]
  var
    img = img
    pad_img = zeros[uint8]([img.shape[0], i_h+2*padding, i_w+2*padding])
    kernel = newTensor[float](ksize, ksize)
    k_w = kernel.shape[1]
    k_h = kernel.shape[0]
    output_w = (i_w + 2*padding - k_w) div stride + 1
    output_h = (i_h + 2*padding - k_h) div stride + 1

  result = newTensor[uint8]([img.shape[0], output_h, output_w])

  for c in 0..<img.shape[0]:
    for i in 0..<i_h:
      for j in 0..<i_w:
        pad_img[c, i+padding, j+padding] = img[c, i, j]

  img = pad_img
  for c in 0..<result.shape[0]:
    for i in 0..<result.shape[1]:
      for j in 0..<result.shape[2]:
        for x in 0..<k_h:
          for y in 0..<k_w:
            kernel[x, y] = exp(-float(x^2+y^2)/(2*(scale_d^2)) - sqrt(float((img[c, i*stride+(k_h div 2), j*stride+(k_w div 2)]^2)-(img[c, i*stride+x, j*stride+y]^2)))/(2*(scale_r^2)))
        var I_xy = 0.0
        kernel = kernel / kernel.sum
        for x in 0..<k_h:
          for y in 0..<k_w:
            I_xy = I_xy + kernel[x, y] * float(img[c, i*stride+x, j*stride+y])
        result[c, i, j] = uint8(I_xy)

write_png(bilateral(noise_img, ksize = 11, scale_r = 3.0, scale_d = 1.0), "images/lena_bilateral.png")
