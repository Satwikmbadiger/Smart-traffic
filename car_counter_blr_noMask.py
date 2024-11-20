from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np


cap = cv2.VideoCapture(r'C:\Users\anike\OneDrive\Desktop\DevTrack\YOLO BASICS\blr.mp4') #for video

model = YOLO("yolov8n.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball", "bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#mask = cv2.imread(r'C:\Users\anike\OneDrive\Desktop\DevTrack\YOLO BASICS\mask_blr_A_D.png')
#mask2 = cv2.imread(r'C:\Users\anike\OneDrive\Desktop\DevTrack\YOLO BASICS\mask_blr_A_F.png')

#tracking
tracker = Sort(max_age=20 , min_hits=2 , iou_threshold=0.3)
#tracker2 = Sort(max_age=20 , min_hits=2 , iou_threshold=0.3)

#A_D(1)
limits_A_D_A =  [480,70,575,480]
limits_A_D_D =  [365,30,455,30]
totalCount_A_D_A = []
totalCount_A_D_D = []

#A_F(2)
limits_A_F_A =  [480,70,575,480]
limits_A_F_F =  [50,180,50,480]
totalCount_A_F_A = []
totalCount_A_F_F = []

#E_D(3)
limits_E_D_E =  [190,55,70,200]
limits_E_D_D =  [365,30,455,30]
totalCount_E_D_E = []
totalCount_E_D_D = []

#E_B(4)
limits_E_B_E =  [190,55,70,200]
limits_E_B_B =  [670,85,670,230]
totalCount_E_B_E = []
totalCount_E_B_B = []

#C_B(5)
limits_C_B_C = [450,50,640,50]
limits_E_C_B =  [700,65,700,230]
totalCount_C_B_C = []
totalCount_E_C_B = []

while True:
  ##A_D(1)

  success , img = cap.read()
  #imgRegion = cv2.bitwise_and(img,mask)

  results = model(img , stream=True)

  detections = np.empty((0,5))
  
  for r in results:
    boxes = r.boxes
    for box in boxes:
      x1,y1,x2,y2 = box.xyxy[0]
      x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

      w , h   = x2 - x1 , y2 - y1

      conf = math.ceil((box.conf[0] * 100))/100
      
      #class name
      cls =int( box.cls[0])
      currentClass = classNames[cls] 

      if currentClass == 'car' or currentClass == 'bus' or currentClass == 'motorbike' or currentClass == 'truck'  and conf > 0.3:
        #cvzone.putTextRect(img , f'{currentClass}  {conf}',(max(0,x1),max(35 ,y1)) , scale=1, thickness=1 , offset=3)
        #cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)
        currentArray = np.array([x1,y1,x2,y2,conf])
        detections= np.vstack((detections , currentArray))

  
  resultsTracker = tracker.update(detections)
  #A_D(1)
  cv2.line(img,(limits_A_D_A[0],limits_A_D_A[1]) , (limits_A_D_A[2],limits_A_D_A[3]) ,(0,0,255) , 5)
  cv2.line(img,(limits_A_D_D[0],limits_A_D_D[1]) , (limits_A_D_D[2],limits_A_D_D[3]) ,(0,0,255) , 2)

  ##A_F (2)
  cv2.line(img,(limits_A_F_A[0],limits_A_F_A[1]) , (limits_A_F_A[2],limits_A_F_A[3]) ,(0,0,255) , 4)
  cv2.line(img,(limits_A_F_F[0],limits_A_F_F[1]) , (limits_A_F_F[2],limits_A_F_F[3]) ,(0,0,255) , 4)

  #E_D(3)
  cv2.line(img,(limits_E_D_E[0],limits_E_D_E[1]) , (limits_E_D_E[2],limits_E_D_E[3]) ,(0,0,255) , 4)
  cv2.line(img,(limits_E_D_D[0],limits_E_D_D[1]) , (limits_E_D_D[2],limits_E_D_D[3]) ,(0,0,255) , 4)

  #E_B(4)
  cv2.line(img,(limits_E_B_E[0],limits_E_B_E[1]) , (limits_E_B_E[2],limits_E_B_E[3]) ,(0,0,255) , 4)
  cv2.line(img,(limits_E_B_B[0],limits_E_B_B[1]) , (limits_E_B_B[2],limits_E_B_B[3]) ,(0,0,255) , 4)

  #C_B(5)
  cv2.line(img,(limits_E_C_B[0],limits_E_C_B[1]) , (limits_E_C_B[2],limits_E_C_B[3]) ,(0,0,255) , 4)
  cv2.line(img,(limits_C_B_C[0],limits_C_B_C[1]) , (limits_C_B_C[2],limits_C_B_C[3]) ,(0,0,255) , 4)



  for result in resultsTracker:
    x1,y1,x2,y2,ID = result
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    #print(result)  #commenting this , or else the terminal will be flooded with info
    w,h= x2-x1 , y2-y1
    cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=2,colorR=(255,0,255))  
    cvzone.putTextRect(img , f' {int(ID)}',(max(0,x1),max(35 ,y1)) , scale=0.8, thickness=2 , offset=8)  

    cx,cy = x1+w//2 , y1+h//2
    cv2.circle(img,(cx,cy) , 5 , (255,0,255) , cv2.FILLED)

    #A_D(1)
    if limits_A_D_A[2]-10<=cx<= limits_A_D_A[2]+10 or limits_A_D_A[0]-10<=cx<= limits_A_D_A[0]+10 and limits_A_D_A[1] <= cy <=limits_A_D_A[3]:
      if totalCount_A_D_A.count(ID)==0:
        totalCount_A_D_A.append(ID)
        cv2.line(img,(limits_A_D_A[0],limits_A_D_A[1]) , (limits_A_D_A[2],limits_A_D_A[3]) ,(0,255,0) , 5)

    if limits_A_D_D[0]<=cx<= limits_A_D_D[2] and limits_A_D_D[1]-15 <= cy <=limits_A_D_D[1]+15:
      if totalCount_A_D_D.count(ID)==0 and totalCount_A_D_A.count(ID)==1:
        totalCount_A_D_D.append(ID)
        cv2.line(img,(limits_A_D_D[0],limits_A_D_D[1]) , (limits_A_D_D[2],limits_A_D_D[3]) ,(0,255,0) , 5)
    

    #A_F(2)
    if (limits_A_F_A[2]-10<=cx<= limits_A_F_A[2]+10 or limits_A_F_A[0]-10<=cx<= limits_A_F_A[0]+10) and (limits_A_F_A[1] <= cy <=limits_A_F_A[3]):
      if totalCount_A_F_A.count(ID)==0:
        totalCount_A_F_A.append(ID)
        cv2.line(img,(limits_A_F_A[0],limits_A_F_A[1]) , (limits_A_F_A[2],limits_A_F_A[3]) ,(0,255,0) , 5)

    if (limits_A_F_F[0]-15<=cx<= limits_A_F_F[0]+25)  and (limits_A_F_F[1] <= cy <=limits_A_F_F[3]):
      if totalCount_A_F_F.count(ID)==0 and totalCount_A_F_A.count(ID)==1:
        totalCount_A_F_F.append(ID)
        cv2.line(img,(limits_A_F_F[0],limits_A_F_F[1]) , (limits_A_F_F[2],limits_A_F_F[3]) ,(0,255,0) , 5)


    #E_D(3)
    if (limits_E_D_E[2]-10<=cx<= limits_E_D_E[2]+10 or limits_E_D_E[0]-10<=cx<= limits_E_D_E[0]+10) and (limits_E_D_E[1]-10 <= cy <=limits_E_D_E[1]+10 or limits_E_D_E[3]-10 <= cy <=limits_E_D_E[3]+10):
      if totalCount_E_D_E.count(ID)==0:
        totalCount_E_D_E.append(ID)
        cv2.line(img,(limits_E_D_E[0],limits_E_D_E[1]) , (limits_E_D_E[2],limits_E_D_E[3]) ,(0,255,0) , 5)

    if (limits_E_D_D[0]-15<=cx<= limits_E_D_D[0]+25)  and (limits_E_D_D[1] <= cy <=limits_E_D_D[3]):
      if totalCount_E_D_D.count(ID)==0 and totalCount_E_D_E.count(ID)==1:
        totalCount_E_D_D.append(ID)
        cv2.line(img,(limits_E_D_D[0],limits_E_D_D[1]) , (limits_E_D_D[2],limits_E_D_D[3]) ,(0,255,0) , 5)

    #E_B(4)
    if limits_E_B_E[2]-10<=cx<= limits_E_B_E[2]+10 or limits_E_B_E[0]-10<=cx<= limits_E_B_E[0]+10 and limits_E_B_E[1]-10 <= cy <=limits_E_B_E[1]+10 or limits_E_B_E[3]-10 <= cy <=limits_E_B_E[3]+10:
      if totalCount_E_B_E.count(ID)==0:
        totalCount_E_B_E.append(ID)
        cv2.line(img,(limits_E_B_E[0],limits_E_B_E[1]) , (limits_E_B_E[2],limits_E_B_E[3]) ,(0,255,0) , 5)

    if limits_E_B_B[0]-15<=cx<= limits_E_B_B[0]+25  and limits_E_B_B[1]-5 <= cy <=limits_E_B_B[3]+5:
      if totalCount_E_B_B.count(ID)==0 and totalCount_E_B_E.count(ID)==1:
        totalCount_E_B_B.append(ID)
        cv2.line(img,(limits_E_B_B[0],limits_E_B_B[1]) , (limits_E_B_B[2],limits_E_B_B[3]) ,(0,255,0) , 5)

    #C_B(5)
    if limits_C_B_C[0]-15<=cx<= limits_C_B_C[0]+25  and limits_C_B_C[1]-25 <= cy <=limits_C_B_C[3]+10:
      if totalCount_C_B_C.count(ID)==0 and totalCount_E_B_E.count(ID)==0:
        totalCount_C_B_C.append(ID)
        cv2.line(img,(limits_C_B_C[0],limits_C_B_C[1]) , (limits_C_B_C[2],limits_C_B_C[3]) ,(0,255,0) , 5)

    if limits_E_C_B[0]-35<=cx<= limits_E_C_B[0]+35  or limits_E_C_B[2]-35<=cx<= limits_E_C_B[2]+35   and limits_E_C_B[3]-25 <= cy <=limits_E_C_B[3]+25 or  limits_E_C_B[1]-25 <= cy <=limits_E_C_B[1]+25:
      if totalCount_E_C_B.count(ID)==0 and totalCount_E_B_E.count(ID)==0 and totalCount_C_B_C.count(ID)==1:
        totalCount_E_C_B.append(ID)
        cv2.line(img,(limits_E_C_B[0],limits_E_C_B[1]) , (limits_E_C_B[2],limits_E_C_B[3]) ,(0,255,0) , 5)
  
  
  
  cvzone.putTextRect(img , f'Count_A_D:{len(totalCount_A_D_D)}  Count_A_F:{len(totalCount_A_F_F)} Count_E_D:{len(totalCount_E_D_D)} Count_E_B:{len(totalCount_E_B_B)} Count_C_B:{len(totalCount_E_C_B)}',(30 ,450) , scale=1, thickness=2)

  cv2.imshow('image',img)
  #cv2.imshow('imageRegion',imgRegion)
  #cv2.imshow('imageRegion2',imgRegion2)
  cv2.waitKey(1) 