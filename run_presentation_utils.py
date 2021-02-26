import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import math, os
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from Config import DatasetName, InputDataSize
from ImageModification import ImageModification
from Config import InputDataSize
import random
import tensorflow as tf
from skimage.transform import resize


def convert_movie_to_imge(video_address, save_path):
    vidcap = cv2.VideoCapture(video_address)
    sec = 0
    frameRate = 0.03  # millisecond
    count = 1
    hasFrames = True

    while hasFrames:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(save_path + "fr_" + str(count) + ".jpg", image)


def detect_face(detect_face):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img = cv2.imread(detect_face)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30))
    if len(faces) == 0:
        return None
    return faces


def detect_FLP(img_path, save_path, model_path):
    model = tf.keras.models.load_model(model_path)
    img_mod = ImageModification()
    offsets = []
    padding = 5
    for i, file in tqdm(enumerate(os.listdir(img_path))):
        img_addr = img_path + file
        img = np.array(Image.open(img_addr)) / 255.0
        face_img = detect_face(img_addr)
        if face_img is None: continue
        x, y, w, h = face_img[0]
        x = x - padding
        y = y + int(5*padding)
        w = w + int(2*padding)
        h = h + int(2*padding)

        img_cropped = img[y:y+h, x:x+w]
        resized_img = resize(img_cropped, (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                             anti_aliasing=True)

        img_d = np.expand_dims(resized_img, axis=0)
        anno_Pre = model.predict(img_d)[0]
        anno_Pre = img_mod.de_normalized(annotation_norm=anno_Pre)

        img_mod.test_image_print(img_name=save_path + 'fr_' + str(i),
                                 img=resized_img, landmarks=anno_Pre)


        # img_mod.test_image_print(img_name=save_path +'fr_' + str(i) + '_pr' + str(i) + '__',
        #                          img=img, landmarks=anno_Pre, offsets=[x,y])
