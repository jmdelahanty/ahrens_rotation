import numpy as np
import cv2 as cv2
import random
# from zaber_motion import Units
import glob
import os
from time import sleep

def newExpDir(savepath):
    numFolds = len(glob.glob(savepath + '/*/', recursive=True))
    print(numFolds)
    newDir = savepath + 'fish' + str(numFolds+1)
    os.mkdir(newDir)
    return newDir

def selRoi(frame):
    roi = cv2.selectROI("select ROI", frame)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
    cv2.destroyAllWindows()
    x,y,w,h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    print(roi)
    return x,y,w,h
def cropImage(x,y,w,h,frame):
    # right wall
    frame[:,x+w:] = 255
    # left wall
    frame[:,0:x] = 255
    # top wall
    frame[0:y,:] = 255
    # bottom wall
    frame[y+h:,:] = 255
    return frame
def getFishPos(frame,thresh,area,aspRat=None):
    cX,cY,contour_list=track(frame,thresh,area,aspRat)
    if not contour_list:
        pos_pxs=np.array([0,0])
        cv2.putText(frame, "lost fish!", (10, 170), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 0), 1, cv2.LINE_AA)
    else:
        cv2.drawContours(frame, contour_list, -1, (50, 50, 50), 2)
        pos_pxs=np.array([cX,cY])

    cv2.putText(frame, "Blobs:" + str(len(contour_list)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1,
                cv2.LINE_AA)
    return pos_pxs
def track(frame,thresh,areaLims,AR=None):
    img = cv2.GaussianBlur(frame, (7, 7), 0)
    retval, img = cv2.threshold(img, thresh, 255, 0)
    img = cv2.medianBlur(img, 5)
    img = cv2.bitwise_not(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list=[]
    cX=[]
    cY=[]
    print(len(contours))
    for contour in contours:
        area = cv2.contourArea(contour)
        minArea=areaLims[0]
        maxArea=areaLims[1]
        if AR is not None:
            minAR=AR[0]
            maxAR=AR[1]
            center,dim,angle=cv2.minAreaRect(contour)
            if max(dim)==0:
                aspectRatio=0
            else:
                aspectRatio=min(dim)/max(dim)
            if area > minArea and area < maxArea and aspectRatio>minAR and aspectRatio<maxAR:
                contour_list.append(contour)
                mom = cv2.moments(contour)
                # cX is along horizontal axis (camera Y) and cY is along vertical (camera X)
                cX = int(mom["m10"] / mom["m00"])
                cY = int(mom["m01"] / mom["m00"])
        else:
            if area > minArea and area < maxArea:
                contour_list.append(contour)
                mom = cv2.moments(contour)
                cX = int(mom["m10"] / mom["m00"])
                cY = int(mom["m01"] / mom["m00"])
    return cX,cY,contour_list
def saveFrame(frame,out):
    gray = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gray_3c = cv2.merge([gray, gray, gray])
    out.write(gray_3c)
def stage_changePos(axisX,axisY,pos,transformMat=None,offsets=None):
    if transformMat is not None:
        pos = np.matmul(transformMat, pos)
    if offsets is not None:
        pos=pos+offsets
    axisX.move_absolute(pos[0], Units.LENGTH_CENTIMETRES,wait_until_idle=False)
    axisY.move_absolute(pos[1], Units.LENGTH_CENTIMETRES, wait_until_idle=False)
def randMot(lims,axisX,axisY,transMat,offsets):
    xlim1,xlim2,ylim1,ylim2=lims
    randCamX = random.randint(xlim1,xlim2)/10
    randCamY= random.randint(ylim1,ylim2)/10
    camPosRand = np.array([randCamX,randCamY,1])
#only move if both X and Y positions are even (25% of the time)
    if all((10*camPosRand[i])%2==0 for i in range(0,2)):
        stage_changePos(axisX,axisY,camPosRand,transMat,offsets)
    else:
        pass

def expPeriod(count,count2,timings):
    freeEnd,statEnd,randDur,chaseDur,trainReps=timings
    if count<freeEnd:
        expPer='free swim'
    elif count>freeEnd and count<statEnd:
        expPer = 'pre training'
    elif count >= statEnd and count < statEnd + ((randDur + chaseDur) * trainReps) and count2<randDur:
        expPer='random motion'
    elif count >= statEnd and count < statEnd + ((randDur + chaseDur) * trainReps) and count2>=randDur and count2<randDur+chaseDur:
        expPer='chase'
    elif count >= statEnd and count < statEnd + ((randDur + chaseDur) * trainReps) and count2==randDur+chaseDur:
        expPer='random motion'
    elif count>statEnd+((randDur+chaseDur)*trainReps) and count<statEnd+((randDur+chaseDur)*trainReps)+(statEnd-freeEnd):
        expPer='post training'
    elif count>statEnd+((randDur+chaseDur)*trainReps)+(statEnd-freeEnd) and count<2*statEnd+((randDur+chaseDur)*trainReps):
        expPer='free swim'
    elif count==2*statEnd+((randDur+chaseDur)*trainReps):
        expPer='not saving anymore'
    else:
        expPer='in between pers'
    return expPer

def serial_write( x , arduino ):
## sends a string (x) to arduino, reads and
## returns Arduino serial output
    arduino.write(bytes(x, 'utf-8'))
    sleep(0.00005)
    #data = arduino.readline()

