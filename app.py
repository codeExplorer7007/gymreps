import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---- Streamlit page setup ----
st.set_page_config(page_title="Smart Dumbbell / Resistance Band Trainer", layout="wide")

st.markdown(
    "<h1 class='title'>Smart Dumbbell / Resistance Band Trainer (Dual-Arm)</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='description'>Counts reps for left and right arms independently, estimates torque, and provides live form feedback.</p>",
    unsafe_allow_html=True,
)

# ---- Session state init ----
if "left_counter" not in st.session_state:
    st.session_state.left_counter = 0
    st.session_state.right_counter = 0
    st.session_state.left_stage = "down"
    st.session_state.right_stage = "down"
    st.session_state.left_torque = 0.0
    st.session_state.right_torque = 0.0

    st.session_state.left_up_frames = 0
    st.session_state.left_down_frames = 0
    st.session_state.right_up_frames = 0
    st.session_state.right_down_frames = 0

    st.session_state.left_feedback = []
    st.session_state.right_feedback = []


def reset_counters():
    st.session_state.left_counter = 0
    st.session_state.right_counter = 0
    st.session_state.left_stage = "down"
    st.session_state.right_stage = "down"
    st.session_state.left_torque = 0.0
    st.session_state.right_torque = 0.0

    st.session_state.left_up_frames = 0
    st.session_state.left_down_frames = 0
    st.session_state.right_up_frames = 0
    st.session_state.right_down_frames = 0

    st.session_state.left_feedback = []
    st.session_state.right_feedback = []


# -------------------------------
# Sidebar UI
# -------------------------------
st.sidebar.header("Smart Equipment Settings")

equipment_type = st.sidebar.selectbox("Equipment", ("Dumbbell", "Resistance Band"))
load_kg = st.sidebar.slider("Load (kg)", 1.0, 40.0, 5.0, 0.5)
arm_length_cm = st.sidebar.slider("Forearm length (cm)", 20.0, 45.0, 30.0, 0.5)

if st.sidebar.button("Reset Counters"):
    reset_counters()

start_clicked = st.sidebar.button("START")


# -------------------------------
# Layout: Stats + Feedback (Left) | Camera (Right)
# -------------------------------
left_col, right_col = st.columns([1, 2], gap="medium")

with left_col:
    left_stats_placeholder = st.empty()
    right_stats_placeholder = st.empty()
    torque_stats_placeholder = st.empty()
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Form Feedback")
    feedback_left_ph = st.empty()
    feedback_right_ph = st.empty()

with right_col:
    st.subheader("Camera")
    VIDEO_DISPLAY_WIDTH = 650
    video_placeholder = st.empty()


# -------------------------------
# Compact CSS styling
# -------------------------------
st.markdown("""
<style>
body { background-color: #020617; }

.title {
    font-size: 2rem; color: #F9FAFB; font-weight: 800;
}

.stat-card {
    background: #0a1220;
    border-radius: 12px;
    padding: 0.5rem 0.8rem;
    border: 1px solid rgba(96,165,250,0.25);
    margin-bottom: 0.4rem;
}

.stat-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #9CA3AF;
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F9FAFB;
    line-height: 1.0;
}

.stat-sub {
    font-size: 0.8rem;
    color: #D1D5DB;
}

.highlight {
    color: #38BDF8;
    font-weight: 600;
}

.feedback {
    background: #0c182a;
    padding: 6px 8px;
    border-radius: 8px;
    margin-bottom: 6px;
    color: #A7F3D0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ---- Helper Functions ----
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def unit(v):
    v = np.array(v); n = np.linalg.norm(v)
    return v / (n + 1e-8)


def vec(a, b):
    return np.array([b[0] - a[0], b[1] - a[1]])


# ---- Form Evaluation ----
def evaluate_form(landmarks, frame_shape):
    feedback = {"left": [], "right": []}
    h, w = frame_shape[:2]

    def xy(lm): return np.array([lm.x * w, lm.y * h])

    try:
        l_sh, l_el, l_wr = xy(landmarks[11]), xy(landmarks[13]), xy(landmarks[15])
        r_sh, r_el, r_wr = xy(landmarks[12]), xy(landmarks[14]), xy(landmarks[16])
        l_hip, r_hip = xy(landmarks[23]), xy(landmarks[24])

        # Form rules
        l_fore = unit(vec(l_el, l_wr)); r_fore = unit(vec(r_el, r_wr))
        vert = np.array([0.0, -1.0])

        if np.degrees(np.arccos(np.clip(np.dot(l_fore, vert), -1, 1))) > 35:
            feedback["left"].append("Keep left wrist neutral")
        if np.degrees(np.arccos(np.clip(np.dot(r_fore, vert), -1, 1))) > 35:
            feedback["right"].append("Keep right wrist neutral")

        # Elbow tuck (horizontal alignment)
        if abs(l_el[0] - l_sh[0]) / (np.linalg.norm(vec(l_sh, l_hip)) + 1e-6) > 0.6:
            feedback["left"].append("Tuck left elbow")
        if abs(r_el[0] - r_sh[0]) / (np.linalg.norm(vec(r_sh, r_hip)) + 1e-6) > 0.6:
            feedback["right"].append("Tuck right elbow")

        # Full range
        l_angle = calculate_angle(l_sh, l_el, l_wr)
        r_angle = calculate_angle(r_sh, r_el, r_wr)

        if l_angle > 170: feedback["left"].append("Fully extend left arm")
        if l_angle < 25:  feedback["left"].append("Fully curl left arm")
        if r_angle > 170: feedback["right"].append("Fully extend right arm")
        if r_angle < 25:  feedback["right"].append("Fully curl right arm")

    except:
        return feedback

    return feedback


# ---- Main Video Processing Loop ----
def capture_video(load_kg, arm_length_cm):
    cap = cv2.VideoCapture(0)
    arm_len_m = arm_length_cm / 100
    force_N = load_kg * 9.81

    UP, DOWN, HOLD = 40, 150, 3

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = pose.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            try:
                lm = results.pose_landmarks.landmark

                # LEFT
                l_sh = [lm[11].x, lm[11].y]; l_el = [lm[13].x, lm[13].y]; l_wr = [lm[15].x, lm[15].y]
                left_angle = calculate_angle(l_sh, l_el, l_wr)
                st.session_state.left_torque = force_N * arm_len_m * np.sin(np.deg2rad(left_angle))

                if left_angle > DOWN:
                    st.session_state.left_down_frames += 1
                    st.session_state.left_up_frames = 0
                    if st.session_state.left_down_frames >= HOLD:
                        st.session_state.left_stage = "down"
                elif left_angle < UP and st.session_state.left_stage == "down":
                    st.session_state.left_up_frames += 1
                    if st.session_state.left_up_frames >= HOLD:
                        st.session_state.left_stage = "up"
                        st.session_state.left_counter += 1

                # RIGHT
                r_sh = [lm[12].x, lm[12].y]; r_el = [lm[14].x, lm[14].y]; r_wr = [lm[16].x, lm[16].y]
                right_angle = calculate_angle(r_sh, r_el, r_wr)
                st.session_state.right_torque = force_N * arm_len_m * np.sin(np.deg2rad(right_angle))

                if right_angle > DOWN:
                    st.session_state.right_down_frames += 1
                    st.session_state.right_up_frames = 0
                    if st.session_state.right_down_frames >= HOLD:
                        st.session_state.right_stage = "down"
                elif right_angle < UP and st.session_state.right_stage == "down":
                    st.session_state.right_up_frames += 1
                    if st.session_state.right_up_frames >= HOLD:
                        st.session_state.right_stage = "up"
                        st.session_state.right_counter += 1

                # Form feedback
                fdbk = evaluate_form(lm, frame.shape)
                st.session_state.left_feedback = fdbk["left"]
                st.session_state.right_feedback = fdbk["right"]

            except:
                pass

            # Draw skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                )

            # ---- Update Stat Cards ----
            left_stats_placeholder.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Left Arm</div>
                    <div class="stat-value">{st.session_state.left_counter}</div>
                    <div class="stat-sub">
                        Stage: <span class="highlight">{st.session_state.left_stage}</span><br/>
                        Torque: <span class="highlight">{st.session_state.left_torque:.1f} Nm</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            right_stats_placeholder.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Right Arm</div>
                    <div class="stat-value">{st.session_state.right_counter}</div>
                    <div class="stat-sub">
                        Stage: <span class="highlight">{st.session_state.right_stage}</span><br/>
                        Torque: <span class="highlight">{st.session_state.right_torque:.1f} Nm</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            torque_stats_placeholder.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Total Load & Torque</div>
                    <div class="stat-value">{load_kg:.1f} kg</div>
                    <div class="stat-sub">
                        Combined torque:<br/>
                        <span class="highlight">{st.session_state.left_torque + st.session_state.right_torque:.1f} Nm</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # ---- Feedback ----
            lf = "<br/>".join(f"- {m}" for m in st.session_state.left_feedback) or "Good form"
            rf = "<br/>".join(f"- {m}" for m in st.session_state.right_feedback) or "Good form"

            feedback_left_ph.markdown(f"<div class='feedback'><b>Left:</b><br>{lf}</div>", unsafe_allow_html=True)
            feedback_right_ph.markdown(f"<div class='feedback'><b>Right:</b><br>{rf}</div>", unsafe_allow_html=True)

            # ---- Show camera ----
            video_placeholder.image(img, channels="BGR", width=VIDEO_DISPLAY_WIDTH)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# ---- Run capture ----
if start_clicked:
    capture_video(load_kg, arm_length_cm)

