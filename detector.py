import cv2 
import mediapipe as mp
from scipy.spatial import distance as dis

def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
            
    p1 = int(top.x * width), int(top.y * height)
    p2 = int(bottom.x * width), int(bottom.y * height)
    
    return dis.euclidean(p1, p2)

def mouth_aspect_ratio(image, outputs, mouth_top, mouth_bottom, mouth_left, mouth_right):
    landmark = outputs.multi_face_landmarks[0]
            
    topLip = landmark.landmark[mouth_top]
    bottomLip = landmark.landmark[mouth_bottom]
    left = landmark.landmark[mouth_left]
    right = landmark.landmark[mouth_right]
    
    vertical = euclidean_distance(image, topLip, bottomLip)
    horizontal = euclidean_distance(image, left, right)
    
    MARValue = horizontal/vertical
    
    return MARValue
    
def eye_aspect_ratio(image, outputs, eye_top, eye_bottom, eye_left, eye_right):

    landmark = outputs.multi_face_landmarks[0]
          
    top = landmark.landmark[eye_top]
    bottom = landmark.landmark[eye_bottom]
    left = landmark.landmark[eye_left]
    right = landmark.landmark[eye_right]

    vertical_distance = euclidean_distance(image, top, bottom) 
    horizontal_distance = euclidean_distance(image, left, right)
    
    EARvalue = horizontal_distance/vertical_distance

    return EARvalue  

draw = mp.solutions.drawing_utils
face_mesh = mp.solutions.face_mesh


eye_coordinates = {
    "left_eye_top":386,
    "left_eye_bottom":374,
    "left_eye_left":263,
    "left_eye_right":362,
    "right_eye_top":159,
    "right_eye_bottom":145,
    "right_eye_left":133,
    "right_eye_right":33
}

mouth_coordinates = {
    "mouth_upper":13,
    "mouth_lower":14,
    "mouth_left":78,
    "mouth_right":308
}


COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)

face_model = face_mesh.FaceMesh(static_image_mode=False, max_num_faces= 1,
                                min_detection_confidence=0.6, min_tracking_confidence=0.5)


import sys

# arg = sys.argv[1]
# counter = 0
# import os 
# files = list(os.listdir())
# vid_files = []
# for vid in files:
#     if vid.split(".")[1] == "avi":
#         vid_files.append(vid)
# vid_files.sort()
        
# capture = cv2.VideoCapture(vid_files[int(arg)])

capture = cv2.VideoCapture(0)

frame_count = 0
min_frame = 20
min_tolerance = 3.6

i  = 0


while True:
    result, image = capture.read()
    
    if result:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        outputs = face_model.process(image_rgb)

        if outputs.multi_face_landmarks:    

            EARLeft = eye_aspect_ratio(image, outputs, eye_coordinates["left_eye_top"], eye_coordinates["left_eye_bottom"], eye_coordinates["left_eye_left"], eye_coordinates["left_eye_right"])
            EARRight = eye_aspect_ratio(image, outputs, eye_coordinates["right_eye_top"], eye_coordinates["right_eye_bottom"], eye_coordinates["right_eye_left"], eye_coordinates["right_eye_right"])
            ratio = (EARLeft + EARRight)/2.0
            
            
            if ratio > min_tolerance:
                frame_count +=1
            else:
                frame_count = 0
                
            if frame_count > min_frame:
                i +=1
                cv2.putText(image,"Drowsy Alert",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
                print(f"Drowsy Alert: It Seems you are sleeping.. please wake up {i}")

            MAR_ratio = mouth_aspect_ratio(image, outputs, mouth_coordinates["mouth_upper"], mouth_coordinates["mouth_lower"], mouth_coordinates["mouth_left"], mouth_coordinates["mouth_right"])

            if MAR_ratio < 1.8:
                i = i+1
                #Open his mouth
                cv2.putText(image,"Drowsy Alert",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
                print(f"Drowsy Alert: It Seems you are sleeping.. please wake up {i} , ratio : {MAR_ratio:.4f} , eyes : {ratio:.4f}")

        ## Draw the landmarks from mouth on the image
        original_image = image
        new_image = cv2.circle(image, (int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_upper"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_upper"]].y * image.shape[0])), 1, COLOR_RED, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_lower"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_lower"]].y * image.shape[0])), 1, COLOR_RED, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_left"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_left"]].y * image.shape[0])), 1, COLOR_RED, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_right"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[mouth_coordinates["mouth_right"]].y * image.shape[0])), 1, COLOR_RED, -1)
        
        ##draw the landmarks from right eyes on the image
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_top"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_top"]].y * image.shape[0])), 1, COLOR_BLUE, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_bottom"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_bottom"]].y * image.shape[0])), 1, COLOR_BLUE, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_left"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_left"]].y * image.shape[0])), 1, COLOR_BLUE, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_right"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["left_eye_right"]].y * image.shape[0])), 1, COLOR_BLUE, -1)

        ##draw the landmarks from left eyes on the image
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_top"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_top"]].y * image.shape[0])), 1, COLOR_BLUE, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_bottom"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_bottom"]].y * image.shape[0])), 1, COLOR_BLUE, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_left"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_left"]].y * image.shape[0])), 1, COLOR_BLUE, -1)
        new_image = cv2.circle(new_image, (int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_right"]].x * image.shape[1]), int(outputs.multi_face_landmarks[0].landmark[eye_coordinates["right_eye_right"]].y * image.shape[0])), 1, COLOR_BLUE, -1)

        ##put the MAR and EAR as text on the image on the top left corner
        cv2.putText(new_image, f"MAR: {MAR_ratio:.4f}", (image.shape[1]-150,30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
        cv2.putText(new_image, f"EAR: {ratio:.4f}", (image.shape[1]-150,60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
        
        cv2.imshow(f"{vid_files[int(arg)]}", original_image)
        if cv2.waitKey(1) & 255 == 27:
            break
        
        
capture.release()
cv2.destroyAllWindows()

print("Vid Done ")
print("-----------------------------------------------------")