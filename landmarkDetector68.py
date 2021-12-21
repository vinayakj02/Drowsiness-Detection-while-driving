import cv2 
import dlib 
import numpy as np


detector = dlib.get_frontal_face_detector()
p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np 
        
        for i,(x,y) in enumerate(shape):
            cv2.circle(frame,(x,y),1,(0,0,255),-1)
        
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()