import cv2
from mtcnn import MTCNN
from deepface import DeepFace

print("OpenCV OK")
detector = MTCNN()
print("MTCNN OK")

print("DeepFace OK")
