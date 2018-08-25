# -*- coding: utf-8 -*-
from PIL import ImageDraw, ImageFont, Image
from random import randint
import shutil
import cv2
import os
import sys
import prog

def progress_update(now, end):
    if now < end:
        sys.stderr.write('\r' + prog.get_progressber_str(now/end))
        sys.stderr.flush
    else:
        sys.stderr.write('\r' + prog.get_progressber_str(now/end))
        sys.stderr.flush
        sys.stderr.write('\n')
        sys.stderr.flush()


class MakeData:
    def __init__(self):
        self.font_list = []
        for font in os.listdir('fonts'):
            self.font_list.append(os.path.splitext(font)[0])
        with open('train_char_list.txt', 'r', encoding="utf-8") as f:
            self.train_char_list = f.read()
        with open('test_char_list.txt', 'r', encoding="utf-8") as f:
            self.test_char_list = f.read()

    def MakeData(self, datatype, rotate=True):
        dir_name = 'traindata/' if datatype=='train' else 'testdata/'
        if rotate:
            rotate_dir = 'traindatas/' if datatype=='train' else 'testdatas/'
        gray_dir = 'train_gray/' if datatype=='train' else 'test_gray/'
        char_list = self.train_char_list if datatype=='train' else self.test_char_list
        fill_size = 7 if datatype=='train' else 3
        if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
        for font in self.font_list:
            if not os.path.isdir(dir_name+font):
                os.mkdir(dir_name+font)
            dic = 'fonts/'+font+'.ttc'
            font_file = ImageFont.truetype(dic, 23)
            for i, kana in enumerate(char_list):
                image = Image.new('RGB', (28, 28), (0, 0, 0))
                draw = ImageDraw.Draw(image)
                orig = (randint(-3, 3), randint(-3, 3))
                draw.text(orig, kana, (255, 255, 255), font=font_file)
                image.save(dir_name+font+'/'+str(i+1).zfill(fill_size)+'.jpg')
            print('Finished make {} font image: {}'.format(datatype, font))
            if rotate: self.CreateRotateImage(datatype, font, rotate_dir, dir_name)
            self.CreateGrayImage(datatype, font, gray_dir, dir_name=(rotate_dir if rotate else dir_name))
        shutil.rmtree(dir_name)
        if rotate: shutil.rmtree(rotate_dir)
    
    def CreateRotateImage(self, datatype, font, create_dir_name, dir_name):
        fill_size = 7 if datatype=='train' else 3
        image_num = 1
        if not os.path.isdir(create_dir_name):
            os.mkdir(create_dir_name)
        file_num = len(os.listdir(dir_name + font))
        for angle in range(0, 360, 4):
            for num in range(file_num):
                im = Image.open(dir_name+font+'/'+str(num+1).zfill(fill_size)+'.jpg')
                if not os.path.isdir(create_dir_name+font):
                    os.mkdir(create_dir_name+font)
                im_r = im.rotate(angle)
                im_r.save(create_dir_name+font+'/'+str(image_num).zfill(fill_size)+'.jpg')
                image_num += 1
            progress_update(angle/4, 89)
        image_num = 1
        print('Finished make rotate data: {}'.format(font))
        shutil.rmtree(dir_name+font)

    def CreateGrayImage(self, datatype, font, create_dir_name, dir_name):
        fill_size = 7 if datatype=='train' else 3
        if not os.path.isdir(create_dir_name):
            os.mkdir(create_dir_name)
        if not os.path.isdir(create_dir_name+font):
            os.mkdir(create_dir_name+font)
        for num in range(1, len(os.listdir(dir_name+font))+1):
            img = cv2.imread(dir_name+font+'/'+str(num).zfill(fill_size)+'.jpg')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(create_dir_name+font+'/'+str(num).zfill(fill_size)+'.jpg', gray)
            progress_update(num, len(os.listdir(dir_name+font)))
        print('Finished make gray image: {}'.format(font))
        shutil.rmtree(dir_name+font)

if __name__ == '__main__':
    makedata = MakeData()
    makedata.MakeData('train')
    makedata.MakeData('test')