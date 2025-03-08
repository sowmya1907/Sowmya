# Hi there! 👋 I'm Sowmya Guduguntla

## 🚀 Data Science | AI & ML Enthusiast  

I am a **Results-driven Data Science professional** with 3.8 years of experience in **data analysis, predictive modeling, and risk assessment**. Skilled in **Python, R, SQL, Tableau, Power BI, and Excel**, with a proven track record of transforming complex data into actionable insights. Currently pursuing my **MSc in Data Science at GITAM Hyderabad**, I am passionate about **leveraging analytics, Machine Learning, Deep Learning, AI, and Generative AI to drive business decisions and create meaningful impact**.

## 🔹 About Me

- 💻 Skilled in **Python, R, SQL**
- 📊 Experienced in **Power BI, Tableau, Microsoft Excela and Data Visualization**
- 🤖 Passionate about **Machine Learning, Deep Learning, AI and Generative AI**
- 🏆 Awarded **2 Spot Awards & 1 Applause Award** at Deloitte for excellence in risk assessment

### 📜 **Certifications**  
✅ **[Certificate of Excellence in technical project competition organized during Machine Learning and Artificial 
Intelligence Internship. ](#)** – Issued by **ExpertsHub- Industry Skill Development Center**  
✅ **[A Two-Day National workshop on LLMs and Generative AI](#)** – Issued by **Gitam University**  
✅ **[Power BI Specialist](#)** – Issued by **Simplilearn** 

---

# 📌 Featured Project: License Plate Recognition

## Overview

This project is a **Deep Learning-based License Plate Recognition System** that detects and recognizes vehicle license plates from images. It is designed for **traffic surveillance, law enforcement, and parking management systems** by leveraging **Convolutional Neural Networks (CNNs) and Optical Character Recognition (OCR)**.

### 🔹 Features

✔️ **License Plate Detection** using YOLO/Faster R-CNN  
✔️ **Character Recognition** using Tesseract OCR & CRNN  
✔️ **End-to-End Pipeline** for real-time plate recognition  

### 🐍 Python Example: Vehicle Plate Recognition

```python
import cv2
import pytesseract

def detect_license_plate(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply OCR
    text = pytesseract.image_to_string(gray, config='--psm 8')
    print("Detected License Plate Number:", text)
    
    # Show image
    cv2.imshow("License Plate", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_license_plate("vehicle.jpg")
```

---

# 📌 Featured Project: Plant Seedling Classification

## Overview

This project utilizes **Deep Learning and Convolutional Neural Networks (CNNs)** to classify plant seedlings into different species. The goal is to automate plant identification to assist in **agriculture, precision farming, and weed detection**.

### 🔹 Features

✔️ **CNN-based image classification** for plant seedling recognition  
✔️ **Preprocessing pipeline** using OpenCV for image enhancement  
✔️ **Trained on a dataset of 12 plant species** with augmentation techniques  
✔️ **Can be extended for mobile or web-based deployment**  

### 🐍 Python Example: Plant Classification Model

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator

def create_plant_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(12, activation='softmax'))  # 12 plant species
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Initialize model
model = create_plant_model()
```

### 🗃️ SQL Query Example: Retrieve Plant Classification Data

```sql
SELECT species, confidence_score 
FROM classification_results
WHERE confidence_score > 0.85;
```

---

## 📌 Installation

To get started, clone this repository and install dependencies:

```bash
# Clone the repo
git clone https://github.com/yourusername/repository-name.git
cd repository-name

# Install dependencies (if applicable)
pip install -r requirements.txt
```

---

## 📌 Technologies Used

- **Programming:** Python, R, SQL  
- **Data Visualization:** Power BI, Tableau  
- **Machine Learning:** Scikit-Learn, TensorFlow, Keras  
- **Computer Vision:** OpenCV, YOLO, OCR (Tesseract, EasyOCR)  
- **Database:** PostgreSQL, MySQL  

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. Fork the repo  
2. Create a new branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Added feature"`)  
4. Push to branch (`git push origin feature-name`)  
5. Open a Pull Request  

---

## 📬 Let's Connect!

📧 Email: [sowmya190700@gmail.com](mailto:sowmya190700@gmail.com)  
🔗 LinkedIn: [https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/sowmya-guduguntla-977790191/)  
📂 Portfolio: [soogudportfolio.netlify.app](https://soogudportfolio.netlify.app)  



