import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize mediapipe pose module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize video capture
cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None
valid_counter = 0  # Counter for valid reps
invalid_counter = 0  # Counter for invalid reps
total_reps = 0  # Counter for total reps
last_stage_change_time = time.time()  # Track time of stage changes
up_start_time = None  # Track when "up" stage starts
rep_start_time = None  # Track when rep starts
feedback_message = ""  # Store feedback like "Slow down!" or "Go down!"
has_marked_invalid_hold = False  # Flag to prevent multiple invalid increments
is_cycle_active = False  # New: Track ongoing cycle
pending_validity = None  # New: Store validity status until cycle completes

# Time thresholds (in seconds)
MIN_REP_DURATION = 1.5  # Minimum time for a valid rep (down to up to down)
MAX_UP_HOLD = 2.0  # Maximum time to hold "up" position
GRACE_PERIOD = 0.5  # Grace period after "Go down!" message

# Setup mediapipe pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make pose detection
        results = pose.process(image)
    
        # Convert back to BGR for OpenCV rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks and calculate angles
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for right arm (side view)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angle at the elbow
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Display angle
            cv2.putText(image, str(int(angle)), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Curl counter logic
            current_time = time.time()
            if angle > 160:
                if stage != "down":
                    stage = "down"
                    last_stage_change_time = current_time
                    up_start_time = None  # Reset up timer
                    has_marked_invalid_hold = False  # Reset flag when leaving "up"
                    # Set rep start time when entering "down" for full cycle
                    if rep_start_time is None:
                        rep_start_time = current_time
                    # Complete cycle when returning to "down" after "up"
                    if is_cycle_active and rep_start_time is not None:
                        rep_duration = current_time - rep_start_time
                        # Finalize rep validity
                        if rep_duration < MIN_REP_DURATION and pending_validity != "invalid_hold":
                            pending_validity = "invalid_fast"
                            feedback_message = "Slow down!"
                        # Update counters only after cycle completion
                        total_reps += 1
                        counter += 1
                        if pending_validity == "valid":
                            valid_counter += 1
                            print(f"Valid Reps: {valid_counter}")
                        else:
                            invalid_counter += 1
                            if pending_validity == "invalid_partial":
                                feedback_message = "Curl higher!"
                                print(f"Invalid Reps: {invalid_counter} (Partial curl)")
                            elif pending_validity == "invalid_fast":
                                print(f"Invalid Reps: {invalid_counter} (Too fast)")
                            elif pending_validity == "invalid_hold":
                                print(f"Invalid Reps: {invalid_counter} (Held too long)")
                        print(f"Total Reps: {total_reps}")
                        # Reset cycle
                        is_cycle_active = False
                        pending_validity = None
                        rep_start_time = current_time  # Restart for next cycle
            if angle < 30 and stage == "down":
                stage = "up"
                is_cycle_active = True  # Start new cycle
                up_start_time = current_time  # Start tracking up duration
                last_stage_change_time = current_time
                has_marked_invalid_hold = False  # Reset flag for new rep
                # Set initial validity based on angle
                if angle < 25:  # Stricter threshold for valid reps
                    pending_validity = "valid"
                    feedback_message = "Good form!"
                else:  # Invalid if angle is between 25° and 30°
                    pending_validity = "invalid_partial"
                    feedback_message = "Curl higher!"
            
            # Check if holding "up" too long
            if stage == "up" and up_start_time is not None:
                up_duration = current_time - up_start_time
                if up_duration > MAX_UP_HOLD:
                    feedback_message = "Go down!"
                    # Give grace period before marking invalid
                    if up_duration > MAX_UP_HOLD + GRACE_PERIOD and not has_marked_invalid_hold:
                        pending_validity = "invalid_hold"
                        feedback_message = "Go down! Rep invalid."
                        has_marked_invalid_hold = True  # Mark as counted
        
        # Render counter box (larger for visibility)
        cv2.rectangle(image, (0, 0), (350, 200), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, str(counter), (15, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, stage if stage else "None", (150, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        # Display valid, invalid, and total rep counters
        cv2.putText(image, f'Valid: {valid_counter}', (15, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Invalid: {invalid_counter}', (15, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Total: {total_reps}', (200, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        # Display feedback message
        cv2.putText(image, f'Feedback: {feedback_message}', (15, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Show feed
        cv2.imshow('Mediapipe Feed', image)

        # Exit when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()