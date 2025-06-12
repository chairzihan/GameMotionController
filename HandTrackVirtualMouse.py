import numpy as np
import HandTrackingModule as htm
import time
import autopy
import cv2

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 5
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
drag_mode = False

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) == 0:
        cv2.putText(img, "No hand detected", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        continue

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
    
    fingers = detector.fingersUp()
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
    clocX = plocX + (x3 - plocX) / smoothening
    clocY = plocY + (y3 - plocY) / smoothening

    if fingers[1] == 1 and fingers[2] == 0:  # Moving mode
        if drag_mode:
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)  # Release drag
            drag_mode = False
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    
    elif fingers[1] == 1 and fingers[2] == 1:  # Click/Drag mode
        length, img, lineInfo = detector.findDistance(8, 12, img)
        if length < 40:  # Click/Drag
            if not drag_mode:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)  # Start drag
                drag_mode = True
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.move(wScr - clocX, clocY)  # Move while dragging
        elif drag_mode:  # Release if fingers separate
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
            drag_mode = False
    
    plocX, plocY = clocX, clocY
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()