import cv2

# Load pre-trained vehicle cascade classifier
car_cascade = cv2.CascadeClassifier('cars.xml')
if car_cascade.empty():
    raise Exception("Error loading Haar cascade file.")

# Open the video file
video = cv2.VideoCapture('traffic.mp4')
if not video.isOpened():
    raise Exception("Error opening video file.")

# Loop over video frames

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break  # Exit loop if no frames are returned

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles using the cascade
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Vehicle', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the frame
    cv2.imshow('Vehicle Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video and destroy all windows
video.release()
cv2.destroyAllWindows()
