Smart Dumbbell / Resistance Band Trainer (Dual-Arm)

Real-time AI-powered fitness tracking using MediaPipe, OpenCV, and Streamlit.

🏋️ Overview

This project is a smart AI-based workout assistant that uses your device’s camera to:

Detect body pose using MediaPipe

Track bicep curl repetitions for both arms independently

Estimate torque (Nm) based on load + arm length

Provide real-time form correction using rule-based feedback

Display a clean dashboard with reps, stages, torque & feedback

Support dumbbells and resistance bands

This allows users to train at home as if supervised by a digital personal trainer.

🚀 Features
✔️ Dual-arm rep counting

Left and right arm are tracked independently with strict rep validation:

Full elbow extension

Full curl

Held positions for accuracy

✔️ Torque estimation

Torque = Load × forearm length × sin(angle)
Used to estimate muscular effort and track exercise intensity.

✔️ Live form feedback (AI-powered)

The system detects:

Wrist bending

Elbow flaring

Body swinging

Incomplete range of motion

Shoulder/hip instability

Shallow curls or half reps

✔️ Streamlit dark UI

Reps, stages, torque, and feedback are displayed outside the camera feed,
so users don’t need to scroll.

📂 Project Structure
gymreps/
│
├── app.py                 # Main Streamlit application
├── PoseEstimationModule.py   # Placeholder for modular code (optional)
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── .gitignore
└── Jupyter notebooks      # Experimentation files

▶️ How to Run
1. Clone the project
git clone https://github.com/codeExplorer7007/gymreps.git
cd gymreps

2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3. Install requirements
pip install -r requirements.txt

4. Run the app
streamlit run app.py

📸 How it Works (Brief Explanation)
1. MediaPipe Pose

Uses 33 body landmarks, especially:
Shoulder
Elbow
Wrist
Hip

2. Angle Calculation

The elbow angle is computed using the geometric formula:

angle = arccos( (a-b)•(c-b) / |a-b||c-b| )

3. Rep Counting Logic
A rep is counted only when:
Arm is fully extended
Arm is fully curled

Both positions are held for several consecutive frames
This removes false positives.

4. Torque Calculation
Torque = Load (N) × Forearm length (m) × sin(angle)

5. Form Feedback

Rule-based biomechanics checks:
Wrist alignment
Elbow position
Body sway
Shoulder stability
Curl depth

🎯 Why This Project Is Useful

Helps beginners learn proper lifting posture
Prevents injuries by detecting bad form
Replaces expensive smart gym equipment
Good for rehabilitation tracking
Works for dumbbells, resistance bands, and bodyweight curls

🛠️ Tech Stack
Component	Technology
Pose Detection	MediaPipe
Vision Processing	OpenCV
Frontend UI	Streamlit
Maths & Logic	NumPy
Camera Streaming	streamlit-webrtc


📌 Future Improvements
Calorie estimation based on torque × time
Workout history graph & analytics
Sound alerts for reps
Exercise mode selection (shoulder press, lateral raise...)
ML-based form classifier
