from tensorflow import keras
import numpy as np
import cv2
import tensorflow as tf


class Classifier:
    def __init__(self, modelPath):
        self.model=tf.keras.models.load_model(modelPath)
        self.imageSize=64

    def get_prediction(self, imgCrop, hand):
        # img input
        imgResize=cv2.resize(imgCrop, (self.imageSize, self.imageSize))
        img_normalized=imgResize/255.0
        img_reshaped=np.reshape(img_normalized, (1, self.imageSize, self.imageSize, 3))

        # coor input
        landmarks=[]
        for lm in hand["lmList"]:
            coords=[lm[0]/64, lm[1]/64]  # img size = 64*64, coor normalization
            landmarks.extend(coords)
        landmarks=np.array(landmarks).reshape(1, -1)  # reshape as (1, 42)

        # prediction by trained model
        prediction=self.model.predict([img_reshaped, landmarks])
        index=np.argmax(prediction)
        probability=prediction[0][index]

        labels={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'space'}

        return labels[index], probability



