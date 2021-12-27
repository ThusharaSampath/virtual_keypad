from time import sleep
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import cv2
import cvzone
from pynput.keyboard import Controller

keys = [["7", "8", "9"],
        ["4", "5", "6"],
        ["1", "2", "3"],
        [".", "0", "<"]]
finalText = ""

keyboard = Controller()


# def drawAll(img, buttonList):
#     for button in buttonList:
#         x, y = button.position
#         w, h = button.size
#         cvzone.cornerRect(img, (button.position[0], button.position[1], button.size[0], button.size[1]),
#                           20, rt=0)
#         cv2.rectangle(img, button.position, (x + w, y + h), (255, 0, 255), cv2.FILLED)
#         cv2.putText(img, button.text, (x + 20, y + 65),
#                     cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
#     return img
def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.position
        cvzone.cornerRect(imgNew, (button.position[0], button.position[1], button.size[0], button.size[1]), 20, rt=0,
                          colorC=(255, 255, 255), colorR=(255, 255, 255))
        cv2.rectangle(imgNew, button.position, (x + button.size[0], y + button.size[1]), (33, 115, 178), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out


class Button:
    def __init__(self, position, text, size=[85, 85]):
        self.position = position
        self.size = size
        self.text = text


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([120 * j + 50, 120 * i + 50], key))

cap = cv2.VideoCapture(0)
cap.set(3, 1250)
cap.set(4, 800)
detector = HandDetector(detectionCon=0.8, maxHands=1)
debounceInterval = 3
lastDebounce = time.time()
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        # Hand 1
        hand = hands[0]
        lmList1 = hand["lmList"]  # List of 21 Landmark points
        bbox1 = hand["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand['center']  # center of the hand cx,cy
        if lmList1:
            for button in buttonList:
                x, y = button.position
                w, h = button.size
                if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (37, 33, 178), cv2.FILLED)
                    # cv2.putText(img, button.text, (x + 20, y + 65),
                    #             cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, _ = detector.findDistance(lmList1[8], lmList1[12])
                    print("distance: ", l)
                    if l < 31:
                        if (time.time() - lastDebounce > debounceInterval):
                            cv2.rectangle(img, button.position, (x + w, y + h), (186, 32, 32), cv2.FILLED)
                            # cv2.putText(img, button.text, (x + 20, y + 65),
                            #             cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            print("clicking")
                            if (button.text == "<"):
                                finalText = finalText[0:len(finalText) - 1]
                            else:
                                keyboard.press(button.text)
                                finalText += button.text
                            sleep(0.05)
                            lastDebounce = time.time()
                        else:
                            print("Debouncing")

    # Display
    displayLength = len(finalText) * 50
    cv2.rectangle(img, (400, 50), (displayLength + 400, 135), (33, 115, 178), cv2.FILLED)
    cv2.putText(img, finalText, (410, 110), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5)
    img = drawAll(img, buttonList)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
