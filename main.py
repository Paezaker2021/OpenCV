import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
외우기 (기본)
import cv2
img = cv2.imread('data/Dog.jpg') #파일 읽어오기
cv2.imshow('지홍이와 준하', img) #(윈도우 창, 읽어올 파일)
cv2.waitKey()
cv2.destroyAllWindows() #윈도우 창을 끄고 전부 삭제함. 무조건 들어가야 함.
'''
#opencv : https://docs.opencv.org/master/

'''
사각형 출력
img = np.zeros((250,500,3), np.uint8)
img = cv2.rectangle(img, (200, 0), (300, 100), (255,255,255), -1) #컴퓨터는 y축이 위에서부터 시작함. 또한 색 배열이 BGR 순임.
#https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9

'''

'''
비디오(웹캠)
cap = cv2.VideoCapture(0)
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
'''

'''
이진법 등 다양한 기술로 사진 보정(숫자나 문자 인식시 중요함)
img = cv2.imread('data/gradient.png', 0)
_, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV) #뒤집기(인버스)
_, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
_, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
_, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow("Image", img)
cv2.imshow("th1", th1)
cv2.imshow("th2", th2)
cv2.imshow("th3", th3)
cv2.imshow("th4", th4)
cv2.imshow("th5", th5)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
이진법으로 숫자가 판별하기 여러울 때, 가우시안 흑백 등 다른 기술로 알기 쉽게 하는 구문
img = cv2.imread('data/sudoku.png',0)
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 11, 2);
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11, 2);

cv2.imshow("Image", img)
cv2.imshow("THRESH_BINARY", th1)
cv2.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
Canny로 더 선명하게 함.
img = cv2.imread("data/lena.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
canny = cv2.Canny(img, 100, 200)

titles = ['image', 'canny']
images = [img, canny]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
'''

'''
img = cv2.imread('data/opencv-logo.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 70, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#등치선(지도상에서 동일한 값을 가진 점을 연결한 선) 추출 대상 이미지, 등치선 추출 방식, 등치선 결과에 대한 근사치화 방식
print("Number of contours = " + str(len(contours)))
print(contours[0])

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)

cv2.imshow('Image', img)
cv2.imshow('Image GRAY', imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
cap = cv2.VideoCapture('data/vtest.avi')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    dff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(dff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations = 3)
    contours, _  = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, 'Status : {}'.format('Movement'),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow('feed', frame1)

    frame1 = frame2
    ret, frame2 = cap.read()
    
    if cv2.waitKey(40) == 27:
        break

        
cv2.destroyAllWindows()
cap.release()
'''

'''
#등치선을 이용한 도형 맞추기
img = cv2.imread('data/shapes.jpg')
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 220, 255 , cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours))

cv2.imshow("img", img)
for contour in contours:
    approx  = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0,0,0),5)
    x = approx.rabel()[0]
    y= approx.rabel()[1] - 5
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    if len(approx) == 4:
        x1, y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
            cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 10:
        cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

cv2.imshow("shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
#사각형 정면이 되게 돌리기
img = cv2.imread('data/sudoku.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
'''

'''
#얼굴 검출
import cv2
from matplotlib import pyplot as plt
faceCascade= cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

img = cv2.imread('data/lena.jpg')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
faces = faceCascade.detectMultiScale(imgGray,1.1,4)
 
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.show()
'''


'''
#영상이나 웹캠 얼굴 인식(변경 시 cap의 변수를 변경하면 됨)
cap = cv2.VideoCapture('data/samplevideo.mp4')
faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(imgGray, 1.1, 3)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

'''


import sys
filename = 'data/1627960189015.jpg'

img = cv2.imread(filename)

if img is None:
    print('Image load failed!')
    exit()

# Load network

net = cv2.dnn.readNet('data/bvlc_googlenet.caffemodel', 'data/deploy.prototxt')

if net.empty():
    print('Network load failed!')
    exit()

# Load class names

classNames = None
with open('data/classification_classes_ILSVRC2012.txt', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Inference

inputBlob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
net.setInput(inputBlob)
prob = net.forward()

# Check results & Display

out = prob.flatten()
classId = np.argmax(out)
confidence = out[classId]

text = '%s (%4.2f%%)' % (classNames[classId], confidence * 100)
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

