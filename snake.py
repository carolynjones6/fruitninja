import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import webbrowser
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

#def open_snake_game():
    # Open the Snake game in the default web browser
    #snake_game_url = "https://www.google.com/search?q=snake+game"
   # webbrowser.open(snake_game_url)
    # Wait for the game to load (increase time if necessary)
    #time.sleep(1)
   # pyautogui.click(x=423, y=649)  #
    

    #pyautogui.click(x=707, y=677)  en

def start_game():
   #pyautogui.click(x=707, y=677)  #
    pyautogui.press("SPACE")

     
def canned_game_play(recognized_gesture):
    if recognized_gesture == "Open_Palm":
        start_game()
    if recognized_gesture == "Thumb_Up":
        pyautogui.press("w")
        print('up')  # Press 'W' for the Thumbs Up gesture
    elif recognized_gesture == "Thumb_Down":
        pyautogui.press("s")  
        print('down')
        # Press 'S' for the Thumbs Up gesture

         
def custom_game_play(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_cmc = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_CMC]

    # Calculate the change in x and y between the tip and the base
    dx = thumb_tip.x - thumb_cmc.x
 
    
    if dx > 0.05:
        pyautogui.press("d")  

        print("Right")
    elif dx < -0.05:
        pyautogui.press("a")  

        print ("Left")

    # Simulate the key press to start the game (use the Enter key)
    

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

        # Flip the image horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Perform gesture recognition on the image
            result_canned = gesture_recognizer.recognize(mp_image)
            result_cus = hands.process(image_rgb)
            #if unkown?

            # Draw the gesture recognition results on the image
            if result_canned.gestures:
                recognized_gesture = result_canned.gestures[0][0].category_name
                confidence = result_canned.gestures[0][0].score
                
                # if closedd fist open game - can comment this but out and hace it already opened 
                #if recognized_gesture == "Closed_Fist":
                    #open_snake_game()
                    
                
                    #if unknwon -then call custom game play function 
                if(recognized_gesture == "None"):
                    if result_cus.multi_hand_landmarks:
                        for hand_landmarks in result_cus.multi_hand_landmarks:
                          # Draw landmarks for the hand 
                            mp_drawing.draw_landmarks(                                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                                # here is where we would callde a recognize left/right point custom gestures
                            print("unknown")
                            custom_game_play(hand_landmarks)
                else:
                        #else its a canned gesture and we can just call canned game play 
                    canned_game_play(recognized_gesture)
                    
                
            
                    
                # Display recognized gesture and confidence
                cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the resulting image
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
