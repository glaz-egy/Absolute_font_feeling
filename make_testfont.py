import os
import random
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image, ImageFont, ImageDraw

font_list = []
for name in os.listdir('fonts'):
    font_list.append(os.path.splitext(name)[0])
kana_list = 'わをんワヲン'

for font in font_list:
    name = font
    if not os.path.isdir('testdata/'+name):
        os.mkdir('testdata/'+name)
    dic = 'fonts/'+font+'.ttc'
    font = ImageFont.truetype(dic, 23)
    for i, kana in enumerate(kana_list):
        image = Image.new('RGB', (28, 28), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        orig = (0, 0)
        draw.text(orig, kana, (255, 255, 255), font=font)
        image.save('testdata/'+name+'/'+str(i+1)+'.jpg')
    print('Finished make font image: {}'.format(name))