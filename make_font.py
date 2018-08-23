import cv2
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image, ImageFont, ImageDraw

font_list = ('meiryo','YuGothL', )
kana_list = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ'

for font in font_list:
    name = font
    dic = font+'.ttc'
    font = ImageFont.truetype(dic, 50)
    for i, kana in enumerate(kana_list):
        siz = font.getsize(kana)
        image = Image.new('RGB', siz, (0, 0, 0))
        draw = ImageDraw.Draw(image)
        orig = (0, -4)
        draw.text(orig, kana, (255, 255, 255), font=font)
        image.save('dataset/'+name+'/'+str(i).zfill(7)+'.jpg')