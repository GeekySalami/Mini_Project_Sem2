import cv2
import mediapipe as mp
from scipy.spatial import distance as dis
import pygame
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
import time




#plt.style.use('dark_background')
alarm_sound_path = "alarm-1-with-reverberation-30031.mp3"

# Initialize Pygame mixer
pygame.mixer.init()
pygame.mixer.music.load(alarm_sound_path)

def play_alarm_sound():
    global play
    if pygame.mixer.music.get_busy() == 0:
        pygame.mixer.music.play(-1)
        time.sleep(3)


def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
            
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    distance = dis.euclidean(point1, point2)
    return distance




RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]


LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

face_mesh = mp.solutions.face_mesh

face_model = face_mesh.FaceMesh(static_image_mode=False,
                                max_num_faces= 1,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.6)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles




def stop_alarm_sound():
    pygame.mixer.music.stop()


def Eye_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]
            
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    
    top_bottom_dis = euclidean_distance(image, top, bottom)
    
    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    
    left_right_dis = euclidean_distance(image, left, right)
    
    EAR = left_right_dis/ top_bottom_dis
    
    return EAR

thresh = 4.3
frame_check = 20

# Flag to indicate if the detection should stop
stop_detection_flag = False


# Function to perform the eye blink detection
def perform_detection():
    global play
    cap = cv2.VideoCapture(2)
    
    flag = 0
    while True:
        if stop_detection_flag:
            break

        result, image = cap.read()
        image = cv2.flip(image,1)
        imcpy = image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        #print(1)
        if result:
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputs = face_model.process(image)
            #print(2)

            #print(3)

            
            if outputs.multi_face_landmarks:

                for face_landmarks in outputs.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                image = imcpy,
                landmark_list = face_landmarks,
                connections = face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
                    

                ratio_left =  Eye_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
            
            
                ratio_right =  Eye_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
            
            
                ratio = (ratio_left + ratio_right)/2.0

                print(ratio, "::", flag)
                
                #print(4)
            
                if ratio >= thresh:
                    flag +=1
                else:
                    flag = 0

                if flag > frame_check:
                	cv2.putText(imcpy,"ALERT!!",(80,40),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,0,255),2)
                	play_alarm_sound()
                
                else:
                    	stop_alarm_sound()
        else:
            print("Error")
            continue


        
        #print(5)
            
           
        cv2.imshow("FACE MESH", imcpy)
        if (cv2.waitKey(2) & 0xFF == ord ('q')):
            print(6)
            break

    #print(8)
    cap.release()
    cv2.destroyAllWindows()


# Create a Tkinter window
root = tk.Tk()
root.title("Eye Blink Detection")
root.geometry("700x600")

# Create a Frame to hold the centered buttons
button_frame = tk.Frame(root)
button_frame.pack(expand=False, fill='both', padx=260, pady=200)

# Start button
start_button = tk.Button(button_frame, text="Start Detection", command=perform_detection)
start_button.pack()

# Function to handle closing the window
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
