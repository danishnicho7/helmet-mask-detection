import cv2
import pygame
from ultralytics import YOLO
import time
import argparse # Import the argparse library for command-line arguments

# -----------------------------
# CONFIG / GLOBALS
# -----------------------------
CONF_THRESH = 0.55 # lower confidence threshold (tune 0.2-0.35)
FRAME_CONFIRM_REQUIRED = 4   # number of consecutive frames to confirm a state

# State counters (used only in video/stream mode)
face_counter = 0
mask_counter = 0
helmet_counter = 0
no_det_counter = 0

current_state = "unknown"       # one of: "granted", "mask", "helmet", "both", "show_face"
USE_HAAR_FALLBACK = False       # set True to enable fallback detection

# -----------------------------
# INITIALIZE AUDIO + MODEL
# -----------------------------
pygame.mixer.init()

# Load YOLOv8 model (best.pt must be in same folder)
print("Loading model... (this may take a second)")
model = YOLO("best.pt")         # replace with path if different
print("Model loaded. Classes:", model.names)

# Load sound files
try:
    mask_sound = pygame.mixer.Sound("mask.mp3")
except Exception as e:
    print(f"Warning: Could not load mask.mp3: {e}")
    mask_sound = None

try:
    helmet_sound = pygame.mixer.Sound("helmet.mp3")
except Exception as e:
    print(f"Warning: Could not load helmet.mp3: {e}")
    helmet_sound = None

try:
    helmet_and_mask_sound = pygame.mixer.Sound("hm.mp3")
except Exception as e:
    print(f"Warning: Could not load hm.mp3: {e}")
    helmet_and_mask_sound = None

# Flag to track if a sound is playing
sound_playing = False

# -----------------------------
# PROCESS DETECTIONS (SMOOTHED, WITH HELMET) - FINAL VERSION
# -----------------------------
def process_detections(frame, results, is_video_stream=True):
    """
    Stabilized detection logic for helmet + mask + face, and UI drawing.
    """
    global sound_playing, face_counter, mask_counter, helmet_counter, no_det_counter, current_state

    has_mask = False
    has_face = False
    has_helmet = False
    detected_list = []

    # parse detections (logic remains the same)
    if len(results) > 0:
        result = results[0]
        for box in result.boxes:
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
            except Exception:
                continue
            
            if conf < CONF_THRESH:
                continue

            name = model.names.get(cls_id, str(cls_id)).lower()
            detected_list.append((name, round(conf, 2)))
            
            if name == "mask":
                has_mask = True
            elif name == "face":
                has_face = True
            elif name == "helmet":
                has_helmet = True
    
    # --- State Logic (NOW CORRECTLY RE-INTEGRATED FOR VIDEO STREAM) ---
    new_state = current_state

    if is_video_stream:
        # Update counters for smoothing
        mask_counter = mask_counter + 1 if has_mask else 0
        helmet_counter = helmet_counter + 1 if has_helmet else 0
        face_counter = face_counter + 1 if has_face and not has_mask and not has_helmet else 0
        no_det_counter = no_det_counter + 1 if (not detected_list) and not (has_face or has_mask or has_helmet) else 0

        # Decide new state only after consecutive frames confirmation
        if (mask_counter >= FRAME_CONFIRM_REQUIRED) and (helmet_counter >= FRAME_CONFIRM_REQUIRED):
            new_state = "both"
        elif helmet_counter >= FRAME_CONFIRM_REQUIRED:
            new_state = "helmet"
        elif mask_counter >= FRAME_CONFIRM_REQUIRED:
            new_state = "mask"
        elif face_counter >= FRAME_CONFIRM_REQUIRED:
            new_state = "granted"
        elif no_det_counter >= FRAME_CONFIRM_REQUIRED:
            new_state = "show_face"
        
    else: # Image Mode Logic (instantaneous)
        globals().update(face_counter=0, mask_counter=0, helmet_counter=0, no_det_counter=0)
        if has_helmet and has_mask:
             new_state = "both"
        elif has_helmet:
            new_state = "helmet"
        elif has_mask:
            new_state = "mask"
        elif has_face:
            new_state = "granted"
        else:
            new_state = "show_face"


    # --- Audio and UI Drawing ---
    message = ""
    box_color_main = (0, 255, 255) 
    
    # Check for state change and handle audio (ONLY in video stream)
    if new_state != current_state:
        current_state = new_state
        if is_video_stream and sound_playing:
            pygame.mixer.stop()
            sound_playing = False
    
    # Determine the message and color based on the current state
    if current_state == "both":
        message = "ACCESS DENIED: Helmet & Mask Detected"
        box_color_main = (0, 0, 255) # Red
        if is_video_stream and helmet_and_mask_sound and not sound_playing:
            try: helmet_and_mask_sound.play(); globals()['sound_playing'] = True
            except Exception as e: print(f"Error playing hm.mp3: {e}")

    elif current_state == "helmet":
        message = "ACCESS DENIED: Helmet Detected"
        box_color_main = (0, 0, 255) # Red
        if is_video_stream and helmet_sound and not sound_playing:
            try: helmet_sound.play(); globals()['sound_playing'] = True
            except Exception as e: print(f"Error playing helmet.mp3: {e}")

    elif current_state == "mask":
        message = "ACCESS DENIED: Mask Detected"
        box_color_main = (0, 0, 255) # Red
        if is_video_stream and mask_sound and not sound_playing:
            try: mask_sound.play(); globals()['sound_playing'] = True
            except Exception as e: print(f"Error playing mask.mp3: {e}")

    elif current_state == "granted":
        message = "ACCESS GRANTED: Clear Face Detected"
        box_color_main = (0, 255, 0) # Green
    
    elif current_state == "show_face":
        message = "NO DETECTION: Show Face/Gear Clearly"
        box_color_main = (0, 255, 255) # Yellow

    # --- Header Bar and Text Drawing (FIXED) ---
    h_frame, w_frame, _ = frame.shape
    header_height = 12 # Reduced header height for more image space
    
    # Draw a filled black rectangle (bar) at the top of the frame
    cv2.rectangle(frame, (0, 0), (w_frame, header_height), (0, 0, 0), -1) 

    # Calculate text properties (using safe, small fixed size)
    font_scale = 0.27
    font_thickness = 1 

    # Set Y position based on fixed header
    text_x = 10
    text_y = int(header_height * 0.65) # Pushes text down 65% of the bar

    # Draw overlay message
    cv2.putText(frame, message, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color_main, font_thickness, cv2.LINE_AA)

    # --- Draw bounding boxes & labels ---
    if len(results) > 0:
        res = results[0]
        for box in res.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls_id, str(cls_id))
            except Exception:
                continue
            
            if conf < CONF_THRESH:
                continue
            
            if name.lower() == "face":
                box_color = (0, 255, 0) 
            elif name.lower() == "mask":
                box_color = (0, 165, 255)
            elif name.lower() == "helmet":
                box_color = (255, 0, 0)
            else:
                box_color = (255, 255, 255)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"{name} {conf:.2f}"
            
            label_font_scale = 0.5 
            label_thickness = 1
            
            cv2.putText(frame, label, (x1, y1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, box_color, label_thickness, cv2.LINE_AA)

    return frame

# -----------------------------
# INPUT MODE FUNCTIONS
# -----------------------------
def run_webcam_loop():
    """Runs the real-time webcam detection loop."""
    global sound_playing
    
    cap = cv2.VideoCapture(0)
    try: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception: pass
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- WEBCAM MODE ---")
    print("Press 'q' in the window to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, verbose=False)
            annotated = process_detections(frame, results, is_video_stream=True)
            cv2.imshow("Helmet & Mask Detection - YOLOv8 (Webcam)", annotated)

            # Reset sound_playing flag
            if not pygame.mixer.get_busy():
                try: globals()['sound_playing'] = False
                except Exception: pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.stop()
        time.sleep(0.1)

def run_on_image(image_path):
    """Loads a single image, runs detection, and displays the result."""
    print(f"\n--- IMAGE MODE ---")
    print(f"Processing image from: {image_path}")
    
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image from {image_path}. Check the path.")
        return
    
    # Run YOLO model on the original frame
    results = model(frame, verbose=False) 
    
    # Process the results (draws the header bar, boxes, and text)
    annotated = process_detections(frame, results, is_video_stream=False)
    
    # --- FIX: Resizing for Display (Guarantees Aspect Ratio Fit) ---
    MAX_SCREEN_WIDTH = 1200  
    MAX_SCREEN_HEIGHT = 800  
    
    h_orig, w_orig, _ = annotated.shape
    
    # Only scale down if the image is too large for the screen
    if w_orig > MAX_SCREEN_WIDTH or h_orig > MAX_SCREEN_HEIGHT:
        # Calculate the single scale factor needed to fit both dimensions
        scale = min(MAX_SCREEN_WIDTH / w_orig, MAX_SCREEN_HEIGHT / h_orig)
        
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        
        # Resize the image for display
        annotated = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow("Detection Output (Image)", annotated)
    
    print("Detection complete. Press any key to close the image window.")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
# -----------------------------
# MAIN EXECUTION WITH ARGPARSE
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-time Helmet and Mask Detection using YOLOv8.")
    parser.add_argument(
        '--image', 
        type=str, 
        default=None, 
        help='Path to a single image file to run detection on. If not provided, webcam mode starts.'
    )
    args = parser.parse_args()

    # Determine input mode based on argument
    if args.image:
        run_on_image(args.image)
    else:
        run_webcam_loop()

if __name__ == "__main__":
    main()