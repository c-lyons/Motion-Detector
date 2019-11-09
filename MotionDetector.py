import cv2
from datetime import datetime
import pandas as pd

first_frame = None    # creating first_frame var for capturing 1st image
status_list = []      # creating empty list to record change in status for time recording
times = []            # creating list for recording time changes
df = pd.DataFrame()

video = cv2.VideoCapture(0)   # creating video capture object and setting camera

while True:
    # create while loop for video capture
    check, frame = video.read()  # check = bool check, frame =  array rep. of image
    status = 0  # status for time analysis of motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # smoothing frame for better results

    if first_frame is None:  # capturing first frame for ref frame to check motion
        first_frame = gray   # setting first frame to frame
        continue             # go back to start of while loop

    delta = cv2.absdiff(first_frame, gray)  # calculating abs diff between first frame and current
    # selecting threshold levels in delta, i.e. difference level (30) and setting color to black pixel
    # if diff > thresh using binary threshold
    thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]  # is tuple, selecting [1]
    # removing "holes" in image to improve image. No kernel - None
    thresh = cv2.dilate(thresh, None, iterations=2)

    # finding contours in thresh image - using copy of thresh, retrieve external method for ext cont
    # and apply chain approx simple to approximate contours
    (cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # if area of contour < 1000 pixels, check next contour
        if cv2.contourArea(contour) < 5000:
            continue
        status = 1  # if detected, status=1 (for time analysis)
        # applying rect to contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    status_list.append(status)  # appending status to record changes

    if len(status_list) < 2:
        pass
    elif status_list[-1] != status_list[-2]:
            # if change in status to prev iter., record datetime object in times
            times.append(datetime.now())

    #cv2.imshow('Capturing', gray)   # show frames in window
    #cv2.imshow('Delta', delta)      # show frames in window
    cv2.imshow('Detecting', frame)  # show frames in window

    key = cv2.waitKey(1)   # defines wait time to release frame
    if key == ord('q'):    # if user enters 'q' then break while loop
        if status == 1:
            times.append(datetime.now())  # if object detected, record time
        break

for i in range(0, len(times), 2):
    df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

df.to_csv('MotionTimes.csv')

video.release()          # releasing video capture object
cv2.destroyAllWindows()  # destroy window





