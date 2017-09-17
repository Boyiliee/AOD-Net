import os
import numpy as np
import caffe
import sys
from pylab import *
import re
import random
import time
import copy
import matplotlib.pyplot as plt
import cv2
import scipy
import shutil
import csv
from PIL import Image
import datetime


def EditFcnProto(templateFile, height, width):
	with open(templateFile, 'r') as ft:
		template = ft.read()
        print templateFile
        outFile = 'DeployT.prototxt'
        with open(outFile, 'w') as fd:
            fd.write(template.format(height=height,width=width))

def test():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    #caffe.set_mode_cpu();

    info = os.listdir('../data/img');
    imagesnum=0;
    for line in info:
        reg = re.compile(r'(.*?).jpg');
        all = reg.findall(line)
        if (all != []):
            imagename = str(all[0]);
            if (os.path.isfile(r'../data/img/%s.jpg' % imagename) == False):
                continue;
            else:
                imagesnum = imagesnum + 1;
                npstore = caffe.io.load_image('../data/img/%s.jpg'% imagename)
                height = npstore.shape[0]
                width = npstore.shape[1]

                templateFile = 'test_template.prototxt'
                EditFcnProto(templateFile, height, width)


                model='../AOD_Net.caffemodel';

                net = caffe.Net('deployT.prototxt', model, caffe.TEST);
                batchdata = []
                data = npstore
                data = data.transpose((2, 0, 1))
                batchdata.append(data)
                net.blobs['data'].data[...] = batchdata;

                net.forward();

                data = net.blobs['sum'].data[0];
                data = data.transpose((1, 2, 0));
                data = data[:, :, ::-1]

                savepath = '../data/result/' + imagename + '_AOD-Net.jpg'
                cv2.imwrite(savepath, data * 255.0,[cv2.IMWRITE_JPEG_QUALITY, 100])

                print imagename

        print 'image numbers:',imagesnum;

def main():
    test()


if __name__ == '__main__':
    main();


