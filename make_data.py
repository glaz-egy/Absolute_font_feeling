import os
from PIL import Image

font_list = []
for name in os.listdir('fonts'):
    font_list.append(os.path.splitext(name)[0])

kana_list = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨ'

file_num = 1

for font in font_list:
    for angle in range(0, 360, 4):
        for num, kana in enumerate(kana_list):
            im = Image.open('dataset/'+font+'/'+str(num+1).zfill(7)+'.jpg')
            if not os.path.isdir('datas/'+font):
                os.mkdir('datas/'+font)
            im_r = im.rotate(angle)
            im_r.save('datas/'+font+'/'+str(file_num).zfill(7)+'.jpg')
            file_num += 1
    file_num = 1
    print('Finished make rotate data: {}'.format(font))