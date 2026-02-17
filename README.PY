

---

# 👁️ Face & Eye Detection using HAAR Cascade (OpenCV)

## 📌 Project Overview

This project detects **faces and eyes** in an image using **Haar Cascade Classifiers** with OpenCV.

It first detects faces in the image and then detects eyes inside each detected face region. Rectangles are drawn around both faces and eyes.

This is a classic **Computer Vision** project and a strong beginner-friendly implementation.

---

## 🛠 Technologies Used

* Python 🐍
* OpenCV
* NumPy
* HAAR Cascade Classifiers

---

## 🧠 How It Works

1. Load Haar cascade XML files for:

   * Face detection
   * Eye detection
2. Load input image.
3. Convert image to grayscale.
4. Detect faces using `detectMultiScale()`.
5. For each detected face:

   * Define ROI (Region of Interest).
   * Detect eyes inside the face region.
6. Draw rectangles:

   * Pink rectangle → Face
   * Yellow rectangle → Eyes
7. Display final output image.

---

## 📂 Project Structure

```
Face-Eye-Detection/
│
├── face_eye_detection.py
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── README.md
```

---

## ⚙️ Installation

### 1️⃣ Install Required Libraries

```bash
pip install opencv-python numpy
```

### 2️⃣ Download Haar Cascade Files

Download from OpenCV official GitHub:

* `haarcascade_frontalface_default.xml`
* `haarcascade_eye.xml`

Place them in your project folder.

---

## ▶️ How to Run

```bash
python face_eye_detection.py
```

---

## 📸 Output

* Detects face(s) in the image
* Detects eyes inside each face
* Draws:

  * 🟣 Pink box → Face
  * 🟡 Yellow box → Eyes
* Displays result in a new window

---

## 💻 Code Explanation (Key Part)

```python
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
    
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    eyes = eye_classifier.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
```

---

## 🎯 Features

✔ Detects multiple faces
✔ Detects multiple eyes per face
✔ Beginner-friendly
✔ Lightweight and fast
✔ Easy to extend to real-time webcam detection

---

## 🚀 Future Improvements

* Real-time webcam detection
* Smile detection
* Emotion detection
* Face recognition
* Mask detection

---

## 👩‍💻 Author

**Ashley**
