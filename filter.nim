import math, strutils, sequtils, random, typetraits, future, macros, os

import arraymancer
var
  img = read_image("images/lena.png") # Read image

proc correlation*(img: Tensor[uint8], kernel: Tensor[int], stride = 1, padding = 0): Tensor[uint8] =
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
    kernel_sum = kernel.sum
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
        var I_xy = 0
        for x in 0..<k_h:
          for y in 0..<k_w:
            inc(I_xy, kernel[x, y] * int(img[c, i*stride+x, j*stride+y]))
        result[c, i, j] = uint8(I_xy div (kernel_sum))

proc smoothing*(img: Tensor[uint8], ksize = 3, stride = 1, padding = 0): Tensor[uint8] =
  var
    kernel = ones[int]([ksize, ksize])
  echo kernel
  result = correlation(img, kernel)
write_png(smoothing(img), "images/lena_smooth.png")
