import cv2
import time

cap = cv2.VideoCapture('demo/input/Fall-1-15-2_Video.avi')
pTime = 200
no_frames = 0
while (cap.isOpened()):
    success, img = cap.read()
    cv2.putText(img, str(int(no_frames)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(pTime) & 0xFF == ord('q'):
        break
    no_frames += 1

cap.release()
cv2.destroyAllWindows()
