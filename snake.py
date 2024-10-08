import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import webbrowser
import time
import pyautogui

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

def open_snake_game():
    # Open the Snake game in the default web browser
    snake_game_url = "https://www.google.com/search?q=snake+game"
    webbrowser.open(snake_game_url)
    # Wait for the game to load (increase time if necessary)
    time.sleep(1)
    pyautogui.click(x=423, y=649)  #
    

    #pyautogui.click(x=707, y=677)  en
def start_game():
    pyautogui.click(x=707, y=677)  #
    

     
def game_play(recognized_gesture):
    if recognized_gesture == "Open_Palm":
        start_game()
    if recognized_gesture == "Thumb_Up":
        pyautogui.press("w")
        print('up')  # Press 'W' for the Thumbs Up gesture
    elif recognized_gesture == "Thumb_Down":
        pyautogui.press("s")  
        print('down')
        # Press 'S' for the Thumbs Up gesture

         
    

    # Simulate the key press to start the game (use the Enter key)
    

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

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
        result = gesture_recognizer.recognize(mp_image)
        #if unkown?

        # Draw the gesture recognition results on the image
        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score
            
            # Example of taking browser action based on recognized gesture
            if recognized_gesture == "Closed_Fist":
                open_snake_game()
                
            else:
                 #if unknwon -then call sustom game play function 
                 game_play(recognized_gesture)
                #move the mouse to the rigth coordinate 
                
                # Make sure to allow for time between recognized gestures so only one window is opened
            
           
                
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
