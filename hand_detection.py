import mediapipe as mp 
import cv2
import numpy as np


class HandDetector():
    
    def __init__(self) -> None:
    
        # Initialize the hand detector and the hand drawer
        self.hand_drawer = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands  
        self.hands_detector = self.mp_hands.Hands()
        
    def find_hands(self, img: np.ndarray, draw:bool=True) -> np.ndarray:
        
        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(imgRGB)
        
        # Draw the landmarks if draw is True
        if draw:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                        self.hand_drawer.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return img
  



        