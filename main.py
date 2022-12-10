from cv2 import cv2


image = cv2.imread('images/FAMILY2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('BASE.xml')
results = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in results:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)


cv2.imshow('Result', image)
cv2.waitKey(0)
