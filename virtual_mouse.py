import cv2 
from hand_detection import HandDetector

vid = cv2.VideoCapture(0)
hand_detector = HandDetector()

while True:
    
    ret, frame = vid.read()
    
    # Drawing detection zone
    frame = cv2.rectangle(frame, (100, 100), (550, 400), (0, 255, 0), 2)
    
    hand_detector.find_hands(frame)
    
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()