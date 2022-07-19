#modul modul yang di butuhkan
import cv2
import os
import numpy as np
from PIL import Image 

recognizer = cv2.face.LBPHFaceRecognizer_create() #algo untuk recognizer yang disediakan OpenCV
detector = cv2.CascadeClassifier(
    "../xml_aset/haarcascade_frontalface_default.xml") #nge detect file harcascade


def getImagesWithLabels(path): #fungsi untuk mempelajari pengenalan gambar dari setiap label (ID)
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] #mengambil data gambar dari folder dan training kesuluruhan data 
    faceSamples = [] #menampung gambar wajah
    Ids = [] #menampung data berdasarkan id
    for imagePath in imagePaths: #lopping untuk mempelajari wajah menggunakan modul PIL yang berguna untuk membuka, memanipulasi dan menyimpan dari berbagai file gambar
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1]) #pengambilan label(id)
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids


faces, Ids = getImagesWithLabels('DataSet') #mendapatkan data wajah dan namanya (ID)di folder Dataset
recognizer.train(faces, np.array(Ids))
recognizer.save('training/training.xml') #disimpan dalam bentuk ekstensi .xml di folder training
