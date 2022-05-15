__authors__ = ['1568205', '1571619', '1571515']
__group__ = 'DM.18'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))



## You can start coding your functions here
def Retrieval_by_color(imatges, etiquetes, colors):
    encontrados = []
    for i, tags in enumerate(etiquetes):
        if all(c in tags for c in colors):
            encontrados.append(imatges[i])
    return encontrados

def Retrieval_by_shape(imatges, etiquetes, classes):
    encontrados = []
    for i, tags in enumerate(etiquetes):
        if all(c in tags for c in classes):
            encontrados.append(imatges[i])
    return encontrados

def Get_color_accuracy(obtenidas, ground_truth):
    accurate= 0
    wrong = []
    for i in range(len(obtenidas)):
        if all(color in set(obtenidas[i]) for color in set(ground_truth[i])):
            accurate+= 1
        else:
            wrong.append(i)
    accuracy = accurate/len(obtenidas)
    return  accuracy, wrong








