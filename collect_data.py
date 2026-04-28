import cv2
import os
import time

print("--- Data Collection ---")
print("Press 's' to save a single image.")
print("Press 'c' to toggle continuous saving (every 0.5s).")
print("Press 'q' to quit.")
label = input("Enter label (e.g. sitting, standing): ")

base_path = os.getcwd()
save_path = os.path.join(base_path, "data", label)

os.makedirs(save_path, exist_ok=True)

print("Saving images to:", save_path)

cap = cv2.VideoCapture(0)
count = 0

# For continuous capture
continuous = False
last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Make a copy for displaying things so we don't save the text on the image
    display_frame = frame.copy()

    # Draw a guide rectangle in the middle to help the user
    height, width, _ = display_frame.shape
    start_point = (int(width * 0.2), int(height * 0.2))
    end_point = (int(width * 0.8), int(height * 0.8))
    cv2.rectangle(display_frame, start_point, end_point, (255, 0, 0), 2)

    # Show info
    mode_text = "Mode: CONTINUOUS" if continuous else "Mode: MANUAL"
    display_text = f"{label} Count: {count} | {mode_text}"
    cv2.putText(display_frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Collecting Data", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        continuous = not continuous
        print("Continuous mode:", continuous)

    if key == ord('s') or (continuous and time.time() - last_save_time > 0.5):
        img_name = os.path.join(save_path, f"{count}.jpg")
        # save the original frame, not the one with the rectangle
        cv2.imwrite(img_name, frame)
        print("Saved:", img_name)
        count += 1
        last_save_time = time.time()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()