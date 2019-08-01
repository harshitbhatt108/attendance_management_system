import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create();
path='TrainingImage'

def getImagewithId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces =[]
    IDs =[]
    for imagePath in imagePaths:
        faceimg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
        return np.array(Ids), faces

    IDs,faces= getImagewithId(path)
    recognizer.train(faces,IDs)
    recognizer.save(r'Trainer/trainer.yml')
    cv2.destroyAllWindows()




                    
