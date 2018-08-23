import cv2
import os

font_list = []
for name in os.listdir('fonts'):
    font_list.append(os.path.splitext(name)[0])

for name in font_list:
    if not os.path.isdir('gray_datas/'+name):
        os.mkdir('gray_datas/'+name)
        os.mkdir('gray_testdata/'+name)
    for num in range(1, len(os.listdir('datas/'+name))+1):
        img = cv2.imread('datas/'+name+'/'+str(num).zfill(7)+'.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray_datas/'+name+'/'+str(num).zfill(7)+'.jpg', gray)

    for num in range(1, len(os.listdir('testdata/'+name))+1):
        img = cv2.imread('testdata/'+name+'/'+str(num)+'.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray_testdata/'+name+'/'+str(num)+'.jpg', gray)
    print('Fnished make gray image: {}'.format(name))