import os
import cv2
import numpy as np
from Classifier_ASL import Classifier
from Hand_Tracking_ASL import HandDetector

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("best_model.h5")
offset=20


def main():
    while True:
        success, img=cap.read()
        hands, img=detector.findHands(img)

        # initialization for variables
        imgCrop=None
        pred_label=""
        pred_prob=0.0

        if hands:
            hand=hands[0]
            x, y, w, h=hand['bbox']

            imgHeight, imgWidth=img.shape[:2]
            y1=max(0, y-offset)
            y2=min(imgHeight, y+h+offset)
            x1=max(0, x-offset)
            x2=min(imgWidth, x+w+offset)

            imgCrop=img[y1:y2, x1:x2]

            if imgCrop.size!=0:  # Check if the crop is not empty
                pred_label, pred_prob=classifier.get_prediction(imgCrop, hand)

                cv2.imshow("ImageCrop", imgCrop)

                large_img=np.zeros((600, 600, 3), dtype=np.uint8)

                center_y=large_img.shape[0]//2
                center_x=large_img.shape[1]//2

                top_y=center_y-imgCrop.shape[0]//2
                top_x=center_x-imgCrop.shape[1]//2

                large_img[top_y:top_y+imgCrop.shape[0], top_x:top_x+imgCrop.shape[1]]=imgCrop

                cv2.putText(large_img, f"Label: {pred_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(large_img, f"Prob: {pred_prob:.2%}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Larger Image", large_img)
                print(f'{pred_label}: {pred_prob:.2%}')

        cv2.putText(img, f"{pred_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"{pred_prob:.2%}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()