import cv2
import numpy as np

from keras.models import model_from_json

json_file = open('classifier_neww.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
myclassifier = model_from_json(loaded_model_json)
myclassifier.load_weights("classifier_new.h5")
print("Loaded model from disk")
myclassifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()
    cv2.rectangle(frame, (237, 170), (337,270), (255,255,255), 2)
    
    subframe = frame[170:270, 237:337]
    gray = cv2.cvtColor(subframe, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(subframe, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    h = cv2.GaussianBlur(h, (5,5),0) 
    _,thresh_v = cv2.threshold(v, 65, 255, cv2.THRESH_BINARY_INV)
    _,thresh_h = cv2.threshold(h, 65, 255, cv2.THRESH_BINARY)
    _,thresh_gray = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    thresh = np.bitwise_or(thresh_h, thresh_gray)
    thresh = np.bitwise_or(thresh, thresh_v)
#    
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas= []
    for c in contours:
        a = cv2.contourArea(c)
        if(a>100):
            areas.append(c)
    thresh = cv2.merge([thresh,thresh,thresh])
    for ar in areas:
        cv2.drawContours(thresh, [ar], -1, (255,255,255), 2)
#    
#    cv2.imshow("video", frame)
#    cv2.imshow('test', frame)
    s = np.reshape(thresh,[1,100,100,3])
    x = str(myclassifier.predict_classes(s)[0])
    cv2.putText(frame, x, (560, 105), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 7 , cv2.LINE_AA)
    cv2.imshow('Digit Recognizer', frame)
    cv2.imshow("sub",thresh)
    ch = cv2.waitKey(1)
    if(ch == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()