from PIL import Image

font_name = 'meiryo'
im = Image.open('dataset/'+font_name+'/00001.jpg')

file_num = 1
for num in range(1, 11):
    im = Image.open('dataset/'+font_name+'/'+str(num).zfill(5)+'.jpg')
    for angle in range(0, 360, 4):
        im_r = im.rotate(angle)
        im_r.save('datas/'+font_name+'/'+str(file_num).zfill(5)+'.jpg')
        file_num += 1