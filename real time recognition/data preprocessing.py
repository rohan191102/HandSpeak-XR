import cv2
import mediapipe as mp
import numpy as np
import os
import random
from tqdm import tqdm
import shutil


class HandProcessor:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.5):
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.offset=10
        self.max_images_per_class=1400  # 1400 img for each class

    def get_bbox_coordinates(self, results, image_shape):
        """Get bounding box coordinates for a hand landmark."""
        all_x, all_y=[], []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks[0].landmark:
                all_x.append(int(handLms.x*image_shape[1]))
                all_y.append(int(handLms.y*image_shape[0]))

            if all_x and all_y:
                return min(all_x), min(all_y), max(all_x), max(all_y)
        return None

    def extract_landmarks(self, results):
        """Extract normalized landmarks from detection results."""
        if results.multi_hand_landmarks:
            landmarks=[]
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y])
            return landmarks
        return None

    def process_image(self, img_path):
        """Process a single image and return cropped image and landmarks."""
        img=cv2.imread(img_path)
        if img is None:
            return None, None

        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results=self.hands.process(imgRGB)

        # get landmarks
        landmarks=self.extract_landmarks(results)
        if landmarks is None:
            return None, None

        # get bbox & crop
        bbox=self.get_bbox_coordinates(results, img.shape)
        if bbox is None:
            return None, None

        minX, minY, MaxX, MaxY=bbox
        h, w=img.shape[:2]
        minX=max(0, minX-self.offset)
        minY=max(0, minY-self.offset)
        MaxX=min(w, MaxX+self.offset)
        MaxY=min(h, MaxY+self.offset)

        imgCrop=img[minY:MaxY, minX:MaxX]
        if imgCrop.size==0:
            return None, None

        return imgCrop, landmarks

    def process_dataset(self, input_path, output_img_path, output_landmarks_path):
        """Process entire dataset and save cropped images and landmarks."""
        os.makedirs(output_img_path, exist_ok=True)
        os.makedirs(output_landmarks_path, exist_ok=True)

        landmarks_dict={}

        for class_name in tqdm(os.listdir(input_path), desc="Processing classes"):
            input_class_path=os.path.join(input_path, class_name)
            if not os.path.isdir(input_class_path):
                continue

            output_class_path=os.path.join(output_img_path, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            # img processing 
            processed_count=0
            for img_name in os.listdir(input_class_path):
                if processed_count>=self.max_images_per_class:
                    break

                input_img_path=os.path.join(input_class_path, img_name)
                cropped_img, landmarks=self.process_image(input_img_path)

                if cropped_img is not None and landmarks is not None:
                    # export cropped image
                    output_img_path_full=os.path.join(output_class_path, img_name)
                    cv2.imwrite(output_img_path_full, cropped_img)

                    # store landmarks
                    relative_path=os.path.join(class_name, img_name)
                    landmarks_dict[relative_path]=landmarks

                    processed_count+=1

        # export landmarks
        np_path=os.path.join(output_landmarks_path, 'hand_landmarks.npz')
        np.savez_compressed(np_path, **landmarks_dict)

        print(f"Processed {len(landmarks_dict)} images")
        print(f"Cropped images saved to: {output_img_path}")
        print(f"Landmarks data saved to: {output_landmarks_path}")


if __name__=="__main__":
    input_path="E:/ML_ASL_try/asl_alphabet_train/asl_alphabet_train/"
    output_img_path="E:/ML_ASL_try/new_cropped_images/"
    output_landmarks_path="E:/ML_ASL_try/new_landmarks_data/"

    processor=HandProcessor()
    processor.process_dataset(input_path, output_img_path, output_landmarks_path)