#!/usr/bin/python
# OpenCV bindings
#python extrai_caracte.py -t treino -l trei.txt > saida_treino
#python extrai_caracte.py -t teste -l tes.txt > saida_teste

import sys
import cv2
# To performing path manipulations
import os
import time
import csv
# For array manipulations
import numpy as np
import argparse as ap
import numpy as np
import re

def HistCodeChain(img):
        width, height =  img.shape
        if width == 0 or height == 0:
                return [0, 0, 0, 0, 0, 0, 0, 0]
        ret,thresh=cv2.threshold(img,127,255,0)
        thresh= abs(thresh - 255)

        #Busca os contornos para calcular o codechain
        if cv2.__version__[0] == "3":
                _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_NONE)
        else:
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_NONE)
        codeChains=[]
        code = [[0, 2, 6], [0, 1, 7], [4, 3, 5]]
        ##Cria os codechains de acordo com os contornos
        for o, objeto in enumerate(contours):
                for i, element in enumerate(objeto):
                        codeChains.append([])
                        if i!= 0:
                                disX = objeto[i][0][1] - objeto[i-1][0][1]
                                disY = objeto[i][0][0] - objeto[i-1][0][0]
                                codeChains[o] = np.append(codeChains[o],
                                                          code[disX][disY])
        ## Calcula a diferenca dos codeChains e repete o experimento
        tamanhoCodeChain = len(codeChains)
        if(tamanhoCodeChain <= 0):
                return [0, 0, 0, 0, 0, 0, 0, 0]
        histogramas = np.zeros((tamanhoCodeChain, 8))
        for i, cc in enumerate(codeChains):
                for j in np.arange(8):
                        histogramas[i][j] = np.count_nonzero(cc == j)
        histograma = np.zeros(8)
        for h in histogramas:
                histograma+=h
        ##Normaliza
        soma = histograma.sum()
        if(soma != 0):
                histograma/=soma
        # Fim da extracao dos contornos
        return histograma.tolist()

def pega_pixel(img, center, x, y):
        new_value = 0
        try:
                if img[x][y] >= center:
                        new_value = 1
        except:
                pass
        return new_value

def calcula_pixel_lbp(img, x, y):
        '''
        64 | 128 |   1
        ----------------
        32 |   0 |   2
        ----------------
        16 |   8 |   4
        '''
        center = img[x][y]
        val_ar = []
        val_ar.append(pega_pixel(img, center, x-1, y+1))     # acima direita
        val_ar.append(pega_pixel(img, center, x, y+1))       # direita
        val_ar.append(pega_pixel(img, center, x+1, y+1))     # abaixo direita
        val_ar.append(pega_pixel(img, center, x+1, y))       # abaixo
        val_ar.append(pega_pixel(img, center, x+1, y-1))     # abaixo esquerda
        val_ar.append(pega_pixel(img, center, x, y-1))       # esquerda
        val_ar.append(pega_pixel(img, center, x-1, y-1))     # acima esquerda
        val_ar.append(pega_pixel(img, center, x-1, y))       # acima
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
                val += val_ar[i] * power_val[i]
        return val




def calc_histogram(image_array):
    
    histogram = np.zeros(256)
    
    for item in image_array:
        histogram[item] += 1
        
    return histogram



def calc_hist(image_array):

    histogram = calc_histogram(image_array)
  
    soma	=	histogram.sum()
        
    if(soma	!=	0):
        
        histogram/=soma
        
    return	histogram


# Get the path of the training set (1e-3)
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
parser.add_argument("-l", "--imageLabels", help="Path to Image Label Files", required="True")
args = vars(parser.parse_args())

# (2e-3)
pasta=(args["trainingSet"])
arquivos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
train_images = [arq for arq in arquivos if arq.lower().endswith(".tif")]
#print (train_images)

# Dictionary containing image paths as keys and corresponding label as value (1e-3)
train_dic = {}
with open(args['imageLabels'], 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
                train_dic[row[0]] = int(row[1])
#print (train_dic)

# List for storing the LBP Histograms, address of images and the corresponding label
X_test = []
X_name = []
y_test = []

# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test (38s)
for train_image in train_images:
        # Read the image
        im = cv2.imread(train_image)
        # Convert to grayscale as LBP works on grayscale image
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        height, width, channel = im.shape
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        lbp = np.zeros((height, width,3), np.uint8)
        for i in range(0, height):#20s
                for j in range(0, width):
                        lbp[i, j] = calcula_pixel_lbp(img, i, j)
        # Calculate the histogram
        hist=calc_hist(lbp)
        #width = width / 2
        #height = height / 2
        histchan = HistCodeChain(img[0:height/2, 0:width/2])#15s->11s
        histchan += HistCodeChain(img[height/2:height, 0:width/2])#15s->11s
        histchan += HistCodeChain(img[0:height/2, width/2:width])#15s->11s
        histchan += HistCodeChain(img[height/2:height, width/2:width])#15s->11s
        #sys.stderr.write(histchan.toList())#print histchan
        histograma = hist.tolist()+histchan
        X_name.append(train_image)
        X_test.append(histograma)
        y_test.append(train_dic[os.path.split(train_image)[1]])
#print (X_name)
#print(X_test)
#print(y_test)


print len(X_test), len(X_test[0])
for i in range (len(X_name)):
        for j in range( len(X_test[i]) ):
                print X_test[i][j],
        print y_test[i]
