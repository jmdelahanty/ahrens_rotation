import cv2 as cv
from utils import *
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

filePath='~/Desktop/'
vidFileName='fullExp.avi'

#save path and names for txt files generated
savePath = filePathUntitled
txtFileName='220712_f14_fullExp.txt'
roiFileName='220712_f14_roi.txt'

#select starting and ending frames
startFrame=0
finalFrame=137000
thresh = 170
area = [ 1,100 ]
#thresh = 200
#area = [50,150]

#aspRat=[0.1,0.4]
dispFrame = False
stagePRE=np.array([0,0])

cap=cv2.VideoCapture(filePath+vidFileName)
count=0
posList=[[0,0]]

#Select roi to be entire chamber. It is important this is consistent across experiments as the fish pos
#will be measured from a coord system with origin at top left of this roi
for i in range(1):
    _,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    roi = selRoi(frame)
    print(roi)
while True:
    ret,frame = cap.read()
    if True:
    #if it is the first frame being read, set it as background
        if count == startFrame:
            background = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            background = background[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        elif count>startFrame and count<finalFrame:
            print('analyzing')
    #convert to gray-scale, crop to roi, subtract background, and invert (so that fish is dark on light background)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
            proc_frame = cv2.subtract(background , frame)
            proc_frame = cv2.bitwise_not(proc_frame)
    #get the fish centroid (in pixels)
            pos_pxs = getFishPos(proc_frame,thresh,area)
        ##keep pos in raw camera coords (origin at top left of roi)
            if not pos_pxs[0] & pos_pxs[1]:
                #pos_orig_coords=posList[-1]
                pos_pxs = posList[-1]
            #else:
            #    pos_orig_coords=pos_pxs+[roi[0],roi[1]]
            #posList.append(pos_orig_coords)
            posList.append(pos_pxs)
            if dispFrame:
                #cv2.circle(frame,(pos_orig_coords[0]-roi[0],pos_orig_coords[1]-roi[1]),3,(0,0,0),-1)
                cv2.circle(frame, (pos_pxs[0] , pos_pxs[1] ), 3, (0, 0, 0), -1)
                cv2.imshow('frame',frame)
                cv2.imshow('processed frame' , proc_frame)
                cv2.waitKey(1)
        elif count>finalFrame:
            print('done')
            break
        else:
            print('not analyzing')
        count += 1
        print(count)
posList.pop(0)
np.savetxt(savePath+txtFileName,posList)
np.savetxt(savePath+roiFileName,roi)
pos=np.array(posList)
plt.plot(pos[1:60000,0],pos[1:60000,1],'--b')
plt.figure()
plt.plot(pos[80000:,0],pos[80000:,1],'--r')
plt.show()