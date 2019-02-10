import math, strutils, sequtils, random, typetraits, future, macros, os

import arraymancer

var
  img = read_image("images/lena.png") # Read image

proc RGB2BGR*(img: Tensor[uint8]): Tensor[uint8] = # RGB->BGR
  result = concat(img[2..2], img[1..1], img[0..0], axis=0)
write_png(RGB2BGR(img), "images/lena_bgr.png")

proc hflip*(img: Tensor[uint8]): Tensor[uint8] = # 上下逆転
  result = img[_, _, ^1..0|-1]
write_png(hflip(img), "images/lena_hflip.png")

proc vflip*(img: Tensor[uint8]): Tensor[uint8] = # 左右逆転
  result = img[_, ^1..0|-1, _]
write_png(vflip(img), "images/lena_vflip.png")

proc vhflip*(img: Tensor[uint8]): Tensor[uint8] = # 上下左右逆転
  result = img[_, ^1..0|-1, ^1..0|-1]
write_png(vhflip(img), "images/lena_vhflip.png")

proc crop*(img: Tensor[uint8]; x, y, width, height: int): Tensor[uint8] = # 切り抜き
  result = img[_, y..<(y+height), x..<(x+width)]
write_png(crop(img, 100, 100, 100, 100), "images/lena_crop.png")

proc center_crop*(img: Tensor[uint8]; width, height: int): Tensor[uint8] = # 中央切り抜き
  let
    x = (img.shape[2] - width) div 2
    y = (img.shape[1] - height) div 2
  result = crop(img, x, y, width, height)
write_png(center_crop(img, 200, 200), "images/lena_center_crop.png")

proc random_crop*(img: Tensor[uint8]; width, height: int): Tensor[uint8] = # ランダム切り抜き
  randomize()
  let
    x = rand(img.shape[2] - width + 1)
    y = rand(img.shape[1] - height + 1)
  result = crop(img, x, y, width, height)
write_png(random_crop(img, 200, 200), "images/lena_random_crop.png")

proc rot90*(img: Tensor[uint8], k: int): Tensor[uint8] = # 90度回転(kは回数)
  case k mod 4:
    of 0:
      result = img
    of 1:
      result = hflip(img.permute(0,2,1))
    of 2:
      result = vhflip(img)
    else:
      result = vflip(img.permute(0,2,1))
write_png(rot90(img, 1), "images/lena_rot90.png")
