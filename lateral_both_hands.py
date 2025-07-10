import cv2
import mediapipe as mp
import numpy as np

# Setup mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Rep counters
valid_reps = 0
wrong_reps = 0
total_reps = 0
stage = None  # None, "start", "up", "down"
feedback = "Waiting for detection..."

# Thresholds for lateral raises (from front view)
MIN_RAISED_THRESHOLD = -0.10  # Wrist must be at least this much higher than shoulder
MAX_RAISED_THRESHOLD = -0.25  # Wrist shouldn't be too high above shoulder (prevents excessive raising)

# Arm raised position (angle) thresholds
MIN_ACCEPTABLE_ANGLE = 165  # Minimum angle for a good lateral raise (degrees)
MAX_ACCEPTABLE_ANGLE = 180  # Maximum angle for a good lateral raise (degrees)

# Start video capture
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    Parameters:
    a, b, c - 2D points (x, y)
    """
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Middle point (elbow)
    c = np.array(c)  # End point (wrist)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure value is within range for arccos
    
    # Calculate angle
    angle = np.arccos(cosine_angle)
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert color and process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Get coordinates for right side
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Get coordinates for left side
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angles
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Get y-difference between shoulders and wrists
                right_diff = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                left_diff = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                
                # Calculate average metrics
                avg_angle = (right_angle + left_angle) / 2
                avg_diff = (right_diff + left_diff) / 2
                
                # Display debug information
                cv2.putText(image, f'Right Angle: {right_angle:.1f}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f'Left Angle: {left_angle:.1f}', (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f'Avg Y-Diff: {avg_diff:.2f}', (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # State machine logic for rep counting
                if stage is None:
                    # Initial position detection - arms down
                    if avg_diff < 0.05:  # Arms are down
                        stage = "start"
                        feedback = "Ready to start!"
                
                elif stage == "start":
                    # Detect upward movement
                    if avg_diff > MIN_RAISED_THRESHOLD:
                        stage = "up"
                        feedback = "Raising arms..."
                
                elif stage == "up":
                    # At top position, evaluate form
                    if avg_diff <= MIN_RAISED_THRESHOLD:
                        # Check if arms are raised in good position
                        if MIN_ACCEPTABLE_ANGLE <= avg_angle <= MAX_ACCEPTABLE_ANGLE and MIN_RAISED_THRESHOLD >= avg_diff >= MAX_RAISED_THRESHOLD:
                            feedback = "✅ Good form! Hold it!"
                        elif avg_diff < MAX_RAISED_THRESHOLD:
                            feedback = "❌ Arms too high!"
                        elif avg_diff > MIN_RAISED_THRESHOLD:
                            feedback = "❌ Raise arms higher!"
                        elif avg_angle < MIN_ACCEPTABLE_ANGLE:
                            feedback = "❌ Arms too far forward!"
                        elif avg_angle > MAX_ACCEPTABLE_ANGLE:
                            feedback = "❌ Arms too far back!"
                        else:
                            feedback = "❌ Adjust arm position!"
                        
                        # Start looking for downward movement
                        stage = "down"
                
                elif stage == "down":
                    # Detect end of rep (arms lowered)
                    if avg_diff < 0.05:
                        total_reps += 1
                        
                        # Evaluate the rep
                        if "✅" in feedback:
                            valid_reps += 1
                        else:
                            wrong_reps += 1
                        
                        # Reset for next rep
                        stage = "start"
                        feedback = f"Rep {total_reps} complete. Ready for next!"
                
            except Exception as e:
                print(f"Error: {e}")
                feedback = "Error tracking pose"
        else:
            feedback = "No pose detected"
        
        # Display feedback and counters
        cv2.putText(image, f'Feedback: {feedback}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if "✅" in feedback else (0, 0, 255), 2)
        
        cv2.putText(image, f'Stage: {stage}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(image, f'Valid Reps: {valid_reps}', (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(image, f'Wrong Reps: {wrong_reps}', (10, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(image, f'Total Reps: {total_reps}', (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Lateral Raise Tracker', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()