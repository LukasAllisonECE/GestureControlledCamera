import cv2
import mediapipe as mp
from fontTools.misc.cython import returns
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from gpiozero import OutputDevice
from time import sleep

mp_hands = mp.tasks.vision.HandLandmarksConnections
#mp_drawing = mp.tasks.vision.drawing_utils
#mp_drawing_styles = mp.tasks.vision.drawing_styles

#Image Annotation parameters, mostly used for testing
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOUR = (88, 205, 54) #Bright green

#Pin out variables
IN1X = OutputDevice(14)
IN2X = OutputDevice(15)
IN3X = OutputDevice(18)
IN4X = OutputDevice(23)

IN1Y = OutputDevice(2)
IN2Y = OutputDevice(3)
IN3Y = OutputDevice(4)
IN4Y = OutputDevice(17)

step_sequence = [
    [1,0,1,0],
    [0,1,1,0],
    [0,1,0,1],
    [1,0,0,1]]


Confidence = 0
cooldown = 0
prevGesture = []
upperBound_x = 160
upperBound_y = 120
lowerBound_x = 480
lowerBound_y = 360
x_pos = 0
y_pos = 0
x_max_step = 85
y_max_step = 290

def step_x(w1,w2,w3,w4):
    IN1X.value = w1
    IN2X.value = w2
    IN3X.value = w3
    IN4X.value = w4
    
def step_y(w1,w2,w3,w4):
    IN1Y.value = w1
    IN2Y.value = w2
    IN3Y.value = w3
    IN4Y.value = w4
    
def step_motorx(steps, direction = 1):
    print("stepx")
    for _ in range(steps):
        for step in (step_sequence if direction >0 else reversed(step_sequence)):
            step_x(*step)  
            sleep(0.01)

def step_motory(steps, direction = 1):
    print("stepy")
    for _ in range(steps):
        for step in (step_sequence if direction >0 else reversed(step_sequence)):
            step_y(*step)
            sleep(0.01)
        
    


def area_detection(recognition_result):
    try:
        if recognition_result.hand_landmarks ==[]:
            x_dir = 0
            y_dir = 0
        else:
            hand_landmarks_list = recognition_result.hand_landmarks

            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                height, width, _ = frame.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                point_x = int((x_coordinates[8]) * width)
                point_y = int((y_coordinates[8]) * height)

                if point_x < upperBound_x:
                    x_dir = 1
                elif point_x > lowerBound_x:
                    x_dir = -1
                else:
                    x_dir = 0
                if point_y < upperBound_y:
                    y_dir = -1
                elif point_y > lowerBound_y:
                    y_dir = 1
                else:
                    y_dir = 0
    except Exception as e:
        print(f"{e}")
        x_dir = 0
        y_dir = 0


    return x_dir, y_dir


base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options,num_hands=1)
recognizer = vision.GestureRecognizer.create_from_options(options)

#Create Video Capture
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
record = False
if not cap.isOpened():
    print("Cannot open camera")
    exit()
print("Camera opened. Press 'q' to exit.")



try:
    while True:
        ret, frame = cap.read()
        cv2.imshow("Webcam Feed",frame)
        if not ret:
            break
        #image processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data = rgb_frame)
        #run vision model
        recognizer_result = recognizer.recognize(mp_image)
        #draw landmarks

        #Error Mitigation
        try:
           if recognizer_result.gestures[0][0].category_name == prevGesture:
               Confidence += 1
           else:
               Confidence = 0
               prevGesture = recognizer_result.gestures[0][0].category_name
        except:
           Confidence = 0
        print (prevGesture)
        print(Confidence)
        print(cooldown)

        area_detection(recognizer_result)

        #Guesture Control
        if Confidence>=7 and cooldown>2:
            cooldown = 0
            try:
                match recognizer_result.gestures[0][0].category_name:
                    case 'Pointing_Up':
                        if record == False:
                            finalImage = cv2.rectangle(frame, (upperBound_x, upperBound_y), (lowerBound_x, lowerBound_y), color=(255, 0, 0), thickness=2)
                            x_dir, y_dir = area_detection(recognizer_result)
                            print(x_dir, y_dir)
                            if x_dir != 0 and ((x_dir*x_pos < 0) or (abs(x_pos) < x_max_step - 35)):
                                step_motorx(35, x_dir)
                                x_pos = x_pos + x_dir*35
                            if y_dir != 0 and ((y_dir*y_pos < 0) or (abs(y_pos) < y_max_step - 100)):
                                step_motory(100,y_dir)
                                y_pos = y_pos + y_dir*100
                            confidence = 0
                    case 'ILoveYou':
                        step_motorx(abs(x_pos), -(x_pos/(abs(x_pos))))
                        step_motory(abs(y_pos), -(y_pos/(abs(y_pos))))
                        confidence = 0
                        break
                    case 'Thumb_Up':
                        record = True
                        finalImage = frame
                    case 'Open_Palm':
                        out.release()
                        record = False
                        finalImage = frame
                    case 'Victory':
                        cv2.imwrite("saved_photo.png",frame)
                        finalImage = frame
                    case _:
                        finalImage = frame
            except:
                finalImage = frame
        else:
            finalImage = frame
        #display image

        if record:
            out.write(finalImage)

        cooldown += 1
        cv2.imshow("Webcam Feed",finalImage)
        


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    #debugging
    print(f"An error occurred during video processing: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()