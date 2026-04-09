import cv2
import os

label = input("Enter label (sitting/standing): ")

base_path = os.getcwd()
save_path = os.path.join(base_path, "data", label)

os.makedirs(save_path, exist_ok=True)

print("Saving images to:", save_path)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_text = f"{label} Count: {count}"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_name = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_name, frame)
        print("Saved:", img_name)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()