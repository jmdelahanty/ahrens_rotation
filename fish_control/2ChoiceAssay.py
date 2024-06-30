from time import sleep, perf_counter
from pymba import Vimba, VimbaException, Frame
from typing import Optional
import cv2
import pyfirmata
from threading import Thread
import os.path
from datetime import datetime
## upload examples>firmata>StandardFirmata from arduino prior to running this python script
## run the script, it will turn on pumps, close valves, and then wait for user to press enter to start the experiment

#path='U:/DZ/Oct 2021/211011'
path = "E:/JD/"
fileBaseName = '20240617_fish'
fileName = fileBaseName + str(1)
fishNum=1
#fileName='testNewChamber'
#path='U:/Weiyu_nearline/2021_Sep/210919'
#fileName='fish' +str(fishNum)+'_60mM_msg_100mM_Sucrose_postLiCl_2'

frameRate=100 #fps

prePeriod = 300 #120 #5 #in seconds
postPeriod = 300 #60 #5 #in seconds
IPI = 0.50 #in seconds
ISI = 60#20#5 #in seconds

numStim = 5#10
numPulses = 1#5

recordingDuration=prePeriod+postPeriod+(2*numStim*IPI*numPulses)+(ISI*numStim)
recordingDuration=recordingDuration
#declare frame counter and valve status flags
frameCount=0
valve1Flag=False
valve2Flag=False
def driveValves(IPI):
    global valve1Flag, valve2Flag
    valve1.write(0)
    valve1Flag = True
    valve2.write(0)
    valve2Flag = True
    sleep(IPI)
    valve1.write(1)
    valve1Flag = False
    valve2.write(1)
    valve2Flag = False
    sleep(IPI)

def saveAndDisplay(frame: Frame, delay: Optional[int] = 1) -> None:
    global frameCount, out, valve1Flag, valve2Flag
    frameCount += 1
    frame = frame.buffer_data_numpy()

    if valve1Flag:
        cv2.putText(frame, "valve 1 OPEN:", (10, 120),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if valve2Flag:
        cv2.putText(frame, "valve 2 OPEN:", (10, 150),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    gray = cv2.normalize(frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gray_3c = cv2.merge([gray, gray, gray])
    out.write(gray_3c)

    cv2.putText(frame, "Frame:" + str(frameCount), (10, 90),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame',frame)
    cv2.waitKey(delay)

def camControl(recordingDuration):
    global startExp
    with Vimba() as vimba:
        camera = vimba.camera(0)
        camera.open()
#set frame rate and exposure time
        feature = camera.feature('AcquisitionFrameRate')
        featureET = camera.feature('ExposureTime')
        feature.value = frameRate
        featureET.value = (1/frameRate)*10**6
# acquire frames
        camera.arm('Continuous', saveAndDisplay)
        # start the experiment
        Thread(target=expControl, daemon=False).start()
        print("started experiment thread")
        camera.start_frame_acquisition()
        sleep(recordingDuration)
        camera.stop_frame_acquisition()
        camera.disarm()

        camera.close()
        out.release()
        print(frameCount)

def expControl():
    global startExp
#pre period
    print("pre stimulus period ---- duration = " +str(prePeriod))
    sleep(prePeriod)
#experimental period
    print("experiment ---- duration = " +str((2*numStim*IPI*numPulses)+(ISI*numStim)))
    tic=perf_counter()
    for i in range(numStim):
        for j in range(numPulses):
            driveValves(IPI)
        sleep(ISI)
    toc=perf_counter()
#post period
    print("post stimulus period ---- duration = " +str(postPeriod))
    sleep(postPeriod)
#turn off pumps
    print('pumps off')
    print(toc-tic)
    pumpIN.write(0)
    pumpOUT.write(0)

if __name__ == "__main__":
    #setup serial connection to arduino, setup pins
    board = pyfirmata.Arduino('COM15')
    pumpIN = board.digital[2]
    pumpOUT = board.digital[3]
    valve1 = board.digital[4]
    valve2 = board.digital[5]

    pumpIN.mode = pyfirmata.OUTPUT
    pumpOUT.mode = pyfirmata.OUTPUT
    valve1.mode = pyfirmata.OUTPUT
    valve2.mode = pyfirmata.OUTPUT
    #turn on pumps, close valves
    #pumpIN.write(0)
    #pumpOUT.write(0)
    #valve1.write(0)
    #valve2.write(0)

    pumpIN.write(1)
    pumpOUT.write(1)
    valve1.write(1)
    valve2.write(1)

    #setup video writer, check if video file already exists
    while os.path.isfile(str(path)+'/'+str(fileName)+'.avi'):
        print("fish number " + str(fishNum) + " video already exists"
                                              "")
        fishNum+=1
        # fileName='fish' +str(fishNum)+'_60mM_msg_60mM_sucrose_postLiCl'
        fileName = fileBaseName + str(fishNum)
    out = cv2.VideoWriter(str(path) + '/' + str(fileName) + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 100, (640, 480))

    #have user press enter to start the recording, start the experiment by starting videoRec thread (exp thread is started in videoRec thread)
    input("press enter to start experiment")
    videoRec = Thread(target=camControl, args=(recordingDuration,), daemon=False)
    videoRec.start()
    print("started video thread")

    sleep(recordingDuration+(0.1*recordingDuration))