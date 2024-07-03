from typing import Optional
import pyfirmata
from threading import Thread, Event, Lock
import os.path
from datetime import datetime
from time import monotonic
import cv2
import queue
import h5py
import numpy as np

# Experiment parameters
prePeriod = 5  # in seconds
postPeriod = 5  # in seconds
IPI = 0.50  # in seconds
ISI = 2  # in seconds

numStim = 1
numPulses = 5

recordingDuration = prePeriod + postPeriod + (2 * numStim * IPI * numPulses) + (ISI * numStim)

# Shared resources
valve1Flag = False
frameCount = 0
start_event = Event()
frame_queue = queue.Queue()
valve1_lock = Lock()
stop_event = Event()

# Timestamp queue for H5 writing
timestamp_queue = queue.Queue()

# Camera frame rate
FRAME_RATE = 20  # fps

def driveValves(IPI):
    global valve1Flag
    with valve1_lock:
        valve1.write(1)
        valve1Flag = True
        timestamp_queue.put(('open', monotonic()))
    end_time = monotonic() + IPI
    while monotonic() < end_time:
        pass
    with valve1_lock:
        valve1.write(0)
        valve1Flag = False
        timestamp_queue.put(('close', monotonic()))
    end_time = monotonic() + IPI
    while monotonic() < end_time:
        pass

def expControl():
    start_event.wait()
    start_time = monotonic()

    # Pre-period
    print("pre stimulus period ---- duration = " + str(prePeriod))
    while monotonic() - start_time < prePeriod:
        pass

    # Experimental period
    print("experiment ---- duration = " + str((2 * numStim * IPI * numPulses) + (ISI * numStim)))
    exp_start = monotonic()
    for _ in range(numStim):
        for _ in range(numPulses):
            driveValves(IPI)
        stim_start = monotonic()
        while monotonic() - stim_start < ISI:
            pass

    # Post-period
    print("post stimulus period ---- duration = " + str(postPeriod))
    post_start = monotonic()
    with valve1_lock:
        valve1.write(0)  # Ensure valve is closed during post-period
    while monotonic() - post_start < postPeriod:
        pass

def camControl(recordingDuration):
    global frameCount
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('experiment.avi', fourcc, FRAME_RATE, (640, 480))

        start_event.wait()

        start_time = monotonic()
        frame_time = start_time
        frame_interval = 1 / FRAME_RATE

        while monotonic() - start_time < recordingDuration and not stop_event.is_set():
            if monotonic() >= frame_time:
                ret, frame = cap.read()
                if ret:
                    frameCount += 1

                    with valve1_lock:
                        if valve1Flag:
                            cv2.putText(frame, "valve 1 OPEN:", (10, 120),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.putText(frame, "Frame:" + str(frameCount), (10, 90),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

                    out.write(frame)
                    frame_queue.put(frame)

                    frame_time += frame_interval
                else:
                    print("Failed to capture frame")
                    break

    except Exception as e:
        print(f"Error in camera control: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        frame_queue.put(None)  # Signal the end of the video

def displayFrames():
    cv2.namedWindow('frame')
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)  # Wait for up to 1 second for a new frame
            if frame is None:
                break
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        except queue.Empty:
            continue  # If no frame is available, continue the loop
    cv2.destroyAllWindows()

def estimate_frame_number(timestamp, start_time):
    elapsed_time = timestamp - start_time
    return int(elapsed_time * FRAME_RATE)

def h5_writer(filename='valve_timestamps.h5'):
    with h5py.File(filename, 'w') as f:
        dt = np.dtype([('event', 'S5'), ('timestamp', 'f8'), ('estimated_frame', 'i4')])
        dset = f.create_dataset('valve_events', (0,), maxshape=(None,), dtype=dt, chunks=True)
        dset.attrs['description'] = 'Valve open and close events with timestamps and estimated frame numbers'
        dset.attrs['timestamp_unit'] = 'seconds since start of experiment'
        dset.attrs['frame_rate'] = FRAME_RATE
        
        buffer = []
        buffer_size = 10  # Adjust this value based on your needs
        
        start_event.wait()  # Wait for the experiment to start
        start_time = monotonic()
        
        while not stop_event.is_set() or not timestamp_queue.empty():
            try:
                event, timestamp = timestamp_queue.get(timeout=1)
                estimated_frame = estimate_frame_number(timestamp, start_time)
                buffer.append((event, timestamp, estimated_frame))
                
                if len(buffer) >= buffer_size:
                    data = np.array(buffer, dtype=dt)
                    dset.resize((dset.shape[0] + len(buffer),))
                    dset[-len(buffer):] = data
                    buffer.clear()
                    f.flush()
            except queue.Empty:
                continue
        
        # Write any remaining data in the buffer
        if buffer:
            data = np.array(buffer, dtype=dt)
            dset.resize((dset.shape[0] + len(buffer),))
            dset[-len(buffer):] = data
            f.flush()

if __name__ == "__main__":
    try:
        # Setup serial connection to Arduino and configure pins
        board = pyfirmata.Arduino('/dev/ttyACM0')
        valve1 = board.digital[4]

        valve1.mode = pyfirmata.OUTPUT
        valve1.write(0)  # Ensure valve is closed before starting experiment

        # Create and start the threads
        exp = Thread(target=expControl, daemon=True)
        cam = Thread(target=camControl, args=(recordingDuration,), daemon=True)
        h5_thread = Thread(target=h5_writer, daemon=True)

        exp.start()
        cam.start()
        h5_thread.start()

        # Trigger the start event to synchronize threads
        input("Press Enter to start experiment")
        start_event.set()

        # Display frames in the main thread
        displayFrames()

        # Signal threads to stop
        stop_event.set()

        # Wait for all threads to complete
        exp.join()
        cam.join()
        h5_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'board' in locals():
            board.exit()
        cv2.destroyAllWindows()