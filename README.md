# AI-powered Real-Time Aircraft Turnaround Monitoring and Delay Prediction System

## 📌 Overview
Aircraft turnaround operations involve multiple time-critical ground handling activities such as passenger deboarding, fueling, baggage handling, and boarding. Delays or unsafe overlaps during this phase can propagate across flight schedules, resulting in operational inefficiencies and financial losses.

This project presents an AI-powered real-time aircraft turnaround monitoring and delay prediction system designed to enhance situational awareness and operational decision-making in airport environments.

Using computer vision (YOLOv8) and machine learning models, the system detects key turnaround activities from video feeds, tracks their durations, identifies unsafe operational overlaps, and predicts activity-level delays before they escalate into full turnaround delays.

The solution integrates real-time activity detection, timestamp extraction, safety alert generation, and predictive analytics into a unified dashboard interface.

<img width="1915" height="912" alt="image1" src="https://github.com/user-attachments/assets/2bed060b-f76e-469f-bdb3-0dafd4054609" />
<img width="1891" height="900" alt="image" src="https://github.com/user-attachments/assets/cf112b5f-e82d-4455-bf80-b565fbd38b43" />


## 🎯 Objectives
- Detect and monitor aircraft turnaround activities using computer vision
- Extract precise timestamps for each ground handling activity
- Predict activity-level and overall turnaround delays using machine learning
- Identify unsafe overlaps between ground operations
- Support data-driven operational decision-making

---

## 🚀 Key Features
- Real-time activity detection using YOLO-based object detection
- Automated activity duration estimation
- Delay prediction using machine learning models
- Safety alert generation for overlapping operations
- Modular and scalable system architecture

---

## 🧠 Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Scikit-learn / Machine Learning models
- Pandas, NumPy
- Flask / Streamlit (if applicable)

---

## 🏗️ System Architecture
The system consists of:
1. Video input and preprocessing module  
2. Computer vision-based activity detection  
3. Feature extraction and timestamping  
4. Delay prediction model  
5. Visualization and reporting layer  

---

## ⚙️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/SuWaddy211/AI-powered-real-time-aircraft-turnaround-monitoring-and-delay-prediction-system.git
   ```

2. Navigate into the folder
   ```bash
   cd AI-powered-real-time-aircraft-turnaround-monitoring-and-delay-prediction-system
   ```

3. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
---

## ▶️ Running the Application

To launch the Streamlit dashboard:
```bash
streamlit run app.py
```
---

## 👥 Contributors
- Su Waddy Tun 
- Zin May Oo
- Anshath Ahamed Ajumil

