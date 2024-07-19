import cv2  # OpenCV
import time  # Delay
import imutils  # Resize

cam = cv2.VideoCapture(0)  # Camera ID
# time.sleep(1)

firstFrame = None
area = 500

while True:
    _, img = cam.read()  # Read from the camera
    text = "Normal"

    img = imutils.resize(img, width=1000)  # Resize

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  # Smoothened

    if firstFrame is None:
        firstFrame = gaussianImg  # Capturing the first frame
        continue
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)  # Absolute difference

    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]

    threshImg = cv2.dilate(threshImg, None, iterations=2)  # Erosion or

    cnts = cv2.findContours(
        threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:  # Check if contour area is greater than the threshold
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
        text = "Moving Object detected"
    print(text)
    cv2.putText(
        img,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )  # Put text on the image
    cv2.imshow("cameraFeed", img)
    key = cv2.waitKey(10)
    print(key)
    if key == ord("a"):
        break

cam.release()
cv2.destroyAllWindows()
