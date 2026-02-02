import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
import joblib
import pandas as pd

# ===========================================================
# 0. MongoDB Setup
# ===========================================================
from pymongo import MongoClient

# 1.1 Connection Setup
@st.cache_resource
def get_db():
    # Replace with your actual connection string if different
    client = MongoClient("mongodb+srv://sutun_db_user:fjXE_B5zwCbzuXx@cluster0.j3i6n2u.mongodb.net/?appName=Cluster0")
    return client["space"]

db = get_db()
turnaround_collection = db["turnaround_logs"]

# 1.2 Helper to format data for Mongo
def save_activity_to_mongo(activity_name, state_dict):
    """Formats and inserts the activity data into MongoDB"""
    try:
        log_entry = {
            "activity": activity_name,
            "start_time": state_dict["start"],
            "end_time": state_dict["end"],
            "duration_seconds": running_duration_s(state_dict),
            "predicted_delay": state_dict["predicted_delay"],
            "timestamp_logged": datetime.now()
        }
        turnaround_collection.insert_one(log_entry)
    except Exception as e:
        st.error(f"MongoDB Log Error: {e}")

# ===========================================================
# 1. CONFIGURATION & CONSTANTS
# ===========================================================
MODEL_PATH = "Models/best_detection.pt"          
DELAY_MODEL_PATH = "Models/aircraft_delay_predictor.pkl"

# Logic thresholds
START_CONFIRM_FRAMES = 10
END_CONFIRM_FRAMES   = 25
MERGE_GAP_SECONDS    = 18
AOI_IOU_TH           = 0.10
PAX_START_COUNT      = 6
PAX_END_COUNT        = 1

# Sides
FUELING_SIDE  = "left"
BAGGAGE_SIDE  = "left"
BOARDING_SIDE = "right"

# Activities List
ACTIVITIES = [
    "Fueling",
    "Baggage Unloading",
    "Baggage Loading",
    "Passenger Deboarding",
    "Passenger Boarding"
]

# Delay Model Logic
EXPECTED_MINUTES = {
    "Fueling": 14.57,
    "Baggage Loading": 11.42,
    "Baggage Unloading": 9.32,
    "Passenger Boarding": 11.17,
    "Passenger Deboarding": 2.51
}

# UI → Model feature name mapping
MODEL_ACTIVITY_MAP = {
    "Fueling": "Fueling",
    "Baggage Loading": "Baggage_Loading",
    "Baggage Unloading": "Baggage_Unloading",
    "Passenger Boarding": "Boarding",
    "Passenger Deboarding": "Deboarding"
}

DELAY_FEATURE_NAMES = [
    'GP_Minute',
    'Fueling_Progress_Ratio', 'Fueling_Progress_Deviation',
    'Baggage_Loading_Progress_Ratio', 'Baggage_Loading_Progress_Deviation',
    'Baggage_Unloading_Progress_Ratio', 'Baggage_Unloading_Progress_Deviation',
    'Boarding_Progress_Ratio', 'Boarding_Progress_Deviation',
    'Deboarding_Progress_Ratio', 'Deboarding_Progress_Deviation'
]

# UI Styling
RIGHT_PANEL_WIDTH = 520
PANEL_BG = (240, 250, 250)  # Light color
TXT_COLOR = (0, 0, 0)  # Black text color
OK_COLOR  = (0, 255, 0)
OFF_COLOR = (150, 150, 150)
ALERT_COLOR = (0, 0, 255)   
SAFE_COLOR = (0, 100, 0)   
SAFE_TXT   = (150, 255, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ===========================================================
# 2. STATE INITIALIZATION
# ===========================================================


if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True

# ===========================================================
# 0. Mongo Setup
# ===========================================================

    # --- ADD THESE TWO LINES HERE ---
    auto_id = datetime.now().strftime("Turnaround_%Y%m%d_%H%M")
    st.session_state['flight_collection_name'] = auto_id

# ===========================================================

    # Logic Phase State
    st.session_state['baggage_phase'] = "UNLOADING"
    st.session_state['pax_phase'] = "DEBOARDING"
    st.session_state['deboarding_completed'] = False
    st.session_state['deboarding_started_once'] = False
    st.session_state['pax_low_counter'] = 0
    st.session_state['gp_detected_first_time'] = None
    # Passenger timing (NEW)
    st.session_state['last_pax_seen_time'] = None
    st.session_state['DEBOARDING_GRACE_SEC'] = 15

    
    # Alert State
    st.session_state['alert_active'] = False
    st.session_state['alert_acknowledged'] = False
    
    # Prediction State
    st.session_state['last_prediction_time'] = 0
    st.session_state['last_debug_features'] = None
    st.session_state['last_prediction_error'] = None
    
    # Activity Tracking (FIXED SYNTAX HERE)
    st.session_state['activity_state'] = {
        a: {
            "active": False, "start": None, "end": None, 
            "start_hits": 0, "end_hits": 0, "last_end": None, 
            "predicted_delay": 0.0
        } for a in ACTIVITIES
    }

# ===========================================================
# 3. CORE LOGIC FUNCTIONS
# ===========================================================
def timestamp(): return datetime.now()
def ts_string(dt): return dt.strftime("%H:%M:%S") if dt else "-"

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a, area_b = (ax2-ax1)*(ay2-ay1), (bx2-bx1)*(by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def draw_transparent_box(img, box, color=(255, 255, 0), alpha=0.25, label=None):
    x1, y1, x2, y2 = map(int, box)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 6)), FONT, 0.55, color, 2)

# AOI Helpers
def fuselage_band(box):
    x1, y1, x2, y2 = box; w = x2 - x1
    return x1 + 0.35 * w, x1 + 0.65 * w

def get_fueling_aoi(box):
    x1, y1, x2, y2 = box; w, h = x2-x1, y2-y1
    if FUELING_SIDE == "left": return (int(x1-0.20*w), int(y1+0.26*h), int(x1+0.18*w), int(y1+0.78*h))
    else: return (int(x2-0.18*w), int(y1+0.26*h), int(x2+0.20*w), int(y1+0.78*h))

def get_cargo_aoi(box):
    x1, y1, x2, y2 = box; w, h = x2-x1, y2-y1
    fus_left, fus_right = fuselage_band(box)
    if BAGGAGE_SIDE == "left": return (int(fus_left-0.32*w), int(y1+0.55*h), int(fus_left-0.02*w), int(y1+0.92*h))
    else: return (int(fus_right+0.02*w), int(y1+0.55*h), int(fus_right+0.32*w), int(y1+0.92*h))

def get_boarding_aois(box):
    x1, y1, x2, y2 = box; w, h = x2-x1, y2-y1
    fus_left, fus_right = fuselage_band(box)
    if BOARDING_SIDE == "right": ax1, ax2 = int(fus_right-0.10*w), int(fus_right+0.28*w)
    else: ax1, ax2 = int(fus_left-0.28*w), int(fus_left+0.10*w)
    return [(ax1, int(y1+0.12*h), ax2, int(y1+0.56*h)), (ax1, int(y1+0.44*h), ax2, int(y1+0.92*h))]

def show_analytical_report():
    st.header(f"📊 Post-Flight Analysis: {st.session_state['flight_collection_name']}")
    
    # Fetch data from the specific collection for this flight
    coll_name = st.session_state['flight_collection_name']
    data = list(db[coll_name].find({}, {"_id": 0}))
    
    if not data:
        st.warning("No data found for this flight yet.")
        return

    df = pd.DataFrame(data)
    
    # 1. Top Level Metrics
    m1, m2, m3 = st.columns(3)
    total_delay = df['delay_prediction'].sum()
    m1.metric("Total Predicted Delay", f"{total_delay:+.1f} min", delta_color="inverse")
    m2.metric("Activities Tracked", len(df))
    m3.metric("Status", "In Progress" if any(df['status'] == "IN_PROGRESS") else "Finalized")

    # 2. Performance Table vs. Benchmarks
    st.subheader("Performance vs. Benchmark")
    
    # Calculate actual minutes from the duration string "MM:SS"
    def get_mins(dur_str):
        m, s = map(int, dur_str.split(':'))
        return m + (s/60)

    report_list = []
    for _, row in df.iterrows():
        act = row['activity']
        actual = get_mins(row['duration'])
        expected = EXPECTED_MINUTES.get(act, 0)
        diff = actual - expected
        
        report_list.append({
            "Activity": act,
            "Actual (Min)": round(actual, 2),
            "Benchmark (Min)": expected,
            "Variance": round(diff, 2),
            "Status": "🔴 Delayed" if diff > 0 else "🟢 On Time"
        })
    
    st.table(pd.DataFrame(report_list))

    # 3. Visual Chart
    st.subheader("Timeline Variance")
    chart_df = pd.DataFrame(report_list)
    st.bar_chart(chart_df.set_index("Activity")["Variance"])
# ===========================================================
# 0. MongoDB Setup
# ===========================================================
# Logic Update
# 1. Improved Helper: This creates the doc immediately when activity starts
def sync_to_mongo(name):
    s = st.session_state['activity_state'][name]
    if s["start"] is None:
        return 

    try:
        # Use the automatically generated ID from session state
        coll_name = st.session_state['flight_collection_name']
        current_collection = db[coll_name]
        
        # Use activity name as the unique filter so we only have ONE row per activity
        filter_criteria = {"activity": name}
        
        updated_data = {
            "$set": {
                "activity": name,
                "start": s["start"],
                "end": s["end"],
                "duration": f"{running_duration_s(s)//60:02}:{running_duration_s(s)%60:02}",
                "delay_prediction": round(s["predicted_delay"], 2),
                "status": "IN_PROGRESS" if s["active"] else "COMPLETED",
                "last_updated": datetime.now()
            }
        }
        
        # Upsert=True prevents the "complexity" by updating the existing row
        current_collection.update_one(filter_criteria, updated_data, upsert=True)
        
    except Exception as e:
        print(f"Mongo Sync Error: {e}")

# 2. Modify update_activity to call the sync
def update_activity(name, is_active_now):
    s = st.session_state['activity_state'][name]
    
    if is_active_now:
        s["start_hits"] += 1; s["end_hits"] = 0
        if not s["active"] and s["start_hits"] >= START_CONFIRM_FRAMES:
            now = timestamp()
            if s["last_end"] and (now - s["last_end"]).total_seconds() <= MERGE_GAP_SECONDS:
                s["active"], s["end"] = True, None
            else:
                s["active"], s["start"], s["end"] = True, now, None
            
            # 1. Sync START to Mongo
            sync_to_mongo(name) 
            
    else:
        s["end_hits"] += 1; s["start_hits"] = 0
        if s["active"] and s["end_hits"] >= END_CONFIRM_FRAMES:
            s["active"] = False
            s["end"] = s["last_end"] = timestamp()
            
            # 2. Sync END to Mongo
            sync_to_mongo(name)

    # 3. Periodic Sync (Updates duration/delay while activity is running)
    # We use a frame counter or simple timer to avoid hitting Mongo every single frame
    if s["active"] and int(time.time()) % 10 == 0: 
        sync_to_mongo(name)
            # --- MODIFIED PART END ---
# ===========================================================

def running_duration_s(s):
    if s["active"] and s["start"]: return int((timestamp() - s["start"]).total_seconds())
    if (not s["active"]) and s["start"] and s["end"]: return int((s["end"] - s["start"]).total_seconds())
    return 0


def draw_dashboard_panel(height):
    panel = np.full((height, RIGHT_PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)

    # Table
    c_act, c_start, c_end, c_dur, c_delay = 10, 200, 300, 380, 450
    y = 90
    cv2.putText(panel, "Activity", (c_act, y), FONT, 0.5, TXT_COLOR, 1)
    cv2.putText(panel, "Start", (c_start, y), FONT, 0.5, TXT_COLOR, 1)
    cv2.putText(panel, "End", (c_end, y), FONT, 0.5, TXT_COLOR, 1)
    cv2.putText(panel, "Dur", (c_dur, y), FONT, 0.5, TXT_COLOR, 1)
    cv2.putText(panel, "Delay", (c_delay, y), FONT, 0.5, TXT_COLOR, 1)
    y += 15
    cv2.line(panel, (c_act, y), (RIGHT_PANEL_WIDTH-10, y), (80, 80, 80), 2)
    y += 30
    
    for a in ACTIVITIES:
        s = st.session_state['activity_state'][a]
        # Freeze color after activity ends
        color = OK_COLOR if s["active"] else OFF_COLOR

        cv2.putText(panel, a, (c_act, y), FONT, 0.55, color, 2)
        cv2.putText(panel, ts_string(s["start"]), (c_start, y), FONT, 0.55, TXT_COLOR, 1)
        cv2.putText(panel, ts_string(s["end"]) if not s["active"] else "-", (c_end, y), FONT, 0.55, TXT_COLOR, 1)
        
        dur = running_duration_s(s)
        cv2.putText(panel, f"{dur//60:02}:{dur%60:02}", (c_dur, y), FONT, 0.55, TXT_COLOR, 1)
        
        pred = s['predicted_delay']
        # Keep the last predicted delay color frozen
        d_color = (0, 0, 255) if pred > 0 else (0, 255, 0)
        cv2.putText(panel, f"{pred:+.1f}", (c_delay, y), FONT, 0.55, d_color, 1)
        y += 40


    # Alert Section Visuals ONLY
    alert_box_top = y + 10 
    if st.session_state['alert_active']:
        # Draw the RED alert box on the image panel
        cv2.rectangle(panel, (0, alert_box_top), (RIGHT_PANEL_WIDTH, alert_box_top + 60), ALERT_COLOR, -1)
        cv2.putText(panel, "ALERT: FUELING + BOARDING!", (40, alert_box_top + 40), FONT, 0.7, (255, 255, 255), 2)
    else:
        # Draw the GREEN safe box on the image panel
        cv2.rectangle(panel, (0, alert_box_top), (RIGHT_PANEL_WIDTH, alert_box_top + 60), SAFE_COLOR, -1)
        cv2.putText(panel, "No Safety Alerts", (10, alert_box_top + 40), FONT, 0.75, SAFE_TXT, 2)

    return panel


# ===========================================================
# 4. PREDICTION LOGIC
# ===========================================================
# ===========================================================
# REVISED PREDICTION LOGIC
# ===========================================================
# ===========================================================
# REVISED PREDICTION LOGIC (With "Freeze" on End)
# ===========================================================
def run_delay_prediction(delay_model):
    gp_min = 0.0
    if st.session_state['gp_detected_first_time']:
        gp_min = (datetime.now() - st.session_state['gp_detected_first_time']).total_seconds() / 60.0

    features = [gp_min]

    order = [
        "Fueling",
        "Baggage Loading",
        "Baggage Unloading",
        "Passenger Boarding",
        "Passenger Deboarding"
    ]

    for ui_name in order:
        s = st.session_state['activity_state'][ui_name]

        curr_min = running_duration_s(s) / 60.0
        exp_min = EXPECTED_MINUTES[ui_name]

        progress_ratio = curr_min / exp_min if exp_min > 0 else 0
        expected_ratio = gp_min / exp_min if exp_min > 0 else 0
        deviation = progress_ratio - expected_ratio

        features.extend([progress_ratio, deviation])

    input_df = pd.DataFrame([features], columns=DELAY_FEATURE_NAMES)
    st.session_state['last_debug_features'] = input_df

    # Get new predictions from model
    preds = delay_model.predict(input_df)[0]

    for i, ui_name in enumerate(order):
        s = st.session_state['activity_state'][ui_name]
        # ONLY update if the activity has not finished yet
        # If s['end'] is not None, it means the activity ended and we "freeze" the value
        if s['end'] is None:
            st.session_state['activity_state'][ui_name]['predicted_delay'] = preds[i]



# 5. STREAMLIT APP
# ===========================================================
# ===========================================================
# 5. STREAMLIT APP UI
# ===========================================================
st.set_page_config(
    layout="wide", 
    page_title="Turnaround AI",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "RED" Generate Report Button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ff4b4b !important;
        color: white !important;
        font-weight: bold;
        border-radius: 5px;
        width: 100%;
    }
    </style>""", unsafe_allow_html=True)

# 1. Model Loading
@st.cache_resource
def load_models():
    # Loading these once saves about 45-60 seconds of "blank screen" time
    yolo = YOLO(MODEL_PATH)
    delay = joblib.load(DELAY_MODEL_PATH)
    return yolo, delay

model, delay_model = load_models()

# 2. Main Dashboard Layout
st.title("✈️ AI Aircraft Turnaround Dashboard")

# ===========================================================
# 5. STREAMLIT APP UI - LAYOUT ORDER
# ===========================================================


# ===========================================================
# 5. STREAMLIT APP UI (TABBED VERSION)
# ===========================================================

# ===========================================================
# 5. STREAMLIT APP UI (HIGH-SPEED TABBED VERSION)
# ===========================================================

# Persistent Camera Connection
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

tab_live, tab_report = st.tabs(["🔴 Live Dashboard", "📊 Analytical Report"])

with tab_live:
    live_container = st.container()
    with live_container:
        alert_placeholder = st.empty()
        # --- ADD THIS BLOCK HERE ---
        if st.session_state.get('alert_active') and not st.session_state.get('alert_acknowledged'):
            with alert_placeholder.container():
                st.error("🚨 SAFETY CRITICAL: Fueling and Boarding active!")
                if st.button("✅ Click to Acknowledge", key="gate_safety_ack", type="primary", use_container_width=True):
                    st.session_state['alert_acknowledged'] = True
                    st.session_state['alert_active'] = False
                    st.rerun()
        # ---------------------------
        image_placeholder = st.empty()



with tab_report:
    st.header("📊 Analytical Report")
    if st.button("🔄 Sync & Refresh Report"):
        st.rerun()

    coll_name = st.session_state['flight_collection_name']
    try:
        data = list(db[coll_name].find({}, {"_id": 0}))
        if data:
            df = pd.DataFrame(data)
            m1, m2, m3 = st.columns(3)
            total_delay = df['delay_prediction'].sum()
            m1.metric("Total Delay Risk", f"{total_delay:+.1f} min", delta_color="inverse")
            m2.metric("Activities Tracked", len(df))
            m3.metric("Status", "Live Syncing")

            st.subheader("Activity Log Details")
            st.dataframe(df, use_container_width=True)

            st.subheader("Delay Prediction Chart")
            st.bar_chart(df.set_index('activity')['delay_prediction'])
        else:
            st.info("No data recorded for this session yet. Stay in the Live tab to process video.")
    except Exception as e:
        st.error(f"MongoDB Fetch Error: {e}")

# ===========================================================
# 6. THE LOOP (Non-Blocking)
# ===========================================================

cap = st.session_state.cap

# We use a standard loop but add a tiny sleep to allow UI responsiveness
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Run YOLO (This is the heavy part)
    results = model(frame, verbose=False)[0]
    annotated = results.plot()
    boxes = results.boxes.xyxy.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)
    names = model.names
    detected = [names[c] for c in clss]
    
    det_boxes = {name: [] for name in names.values()}
    for i, c in enumerate(clss): det_boxes[names[c]].append(boxes[i])
    
    annotated = results.plot()
    
    # AOIs
    ac_boxes = det_boxes.get("aircraft", [])
    f_aoi, c_aoi, b_aois = None, None, []
    if ac_boxes:
        ac_box = max(ac_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        f_aoi = get_fueling_aoi(ac_box)
        c_aoi = get_cargo_aoi(ac_box)
        b_aois = get_boarding_aois(ac_box)
    
    # --- LOGIC ---
    # 1. Ground Power
    if "ground_power" in detected and st.session_state['gp_detected_first_time'] is None:
        st.session_state['gp_detected_first_time'] = datetime.now()
        
    # 2. Fueling
    f_active = False
    if f_aoi:
        f_active = any(iou_xyxy(t, f_aoi) >= AOI_IOU_TH for t in det_boxes.get("fuel_truck", []))
    update_activity("Fueling", f_active)
    
    # 3. Baggage (Using Session State Phase)
    truck_present = len(det_boxes.get("baggage_truck", [])) > 0
    unload_done = st.session_state['activity_state']["Baggage Unloading"]["last_end"] is not None
    
    if st.session_state['baggage_phase'] == "UNLOADING":
        update_activity("Baggage Unloading", truck_present)
        update_activity("Baggage Loading", False)
        if not truck_present and unload_done:
            st.session_state['baggage_phase'] = "WAIT_LOADING_TRUCK"
    elif st.session_state['baggage_phase'] == "WAIT_LOADING_TRUCK":
        update_activity("Baggage Unloading", False)
        update_activity("Baggage Loading", False)
        if truck_present: st.session_state['baggage_phase'] = "LOADING"
    elif st.session_state['baggage_phase'] == "LOADING":
        update_activity("Baggage Unloading", False)
        update_activity("Baggage Loading", truck_present)
        
    # 4. Passenger (Fixed Logic with Session State)
    def box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def point_in_box(pt, box):
        x, y = pt
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    pax_count = 0
    for p in det_boxes.get("person", []):
        cx, cy = box_center(p)
        if any(point_in_box((cx, cy), aoi) for aoi in b_aois):
            pax_count += 1

    if pax_count > 0:
        st.session_state['last_pax_seen_time'] = time.time()



    # ===============================
    # PASSENGER STATE MACHINE (REVISED)
    # ===============================

    now_t = time.time()
    pax_phase = st.session_state['pax_phase']
    last_seen = st.session_state['last_pax_seen_time']
    started_once = st.session_state['deboarding_started_once']
    completed = st.session_state['deboarding_completed']

    # DEBOARDING PHASE
    if pax_phase == "DEBOARDING":
        
        # Deboarding is active if enough pax are present or it already started
        deb_now = (pax_count >= PAX_START_COUNT) or started_once
        update_activity("Passenger Deboarding", deb_now)
        update_activity("Passenger Boarding", False)
        
        # Mark that deboarding has started at least once
        if deb_now:
            st.session_state['deboarding_started_once'] = True

        # Check grace period to end deboarding
        if started_once and last_seen and (now_t - last_seen > st.session_state['DEBOARDING_GRACE_SEC']):
            st.session_state['pax_phase'] = "BOARDING"
            st.session_state['deboarding_completed'] = True
            # Freeze Deboarding activity
            update_activity("Passenger Deboarding", False)

    # BOARDING PHASE
    elif pax_phase == "BOARDING":
        
        # Boarding is active if deboarding is completed and enough pax are present
        board_now = completed and (pax_count >= PAX_START_COUNT)
        update_activity("Passenger Boarding", board_now)
        update_activity("Passenger Deboarding", False)


        
    # --- ALERT ---
# --- INSIDE THE WHILE LOOP ---

# 1. Check for violation
    fuel_act = st.session_state['activity_state']["Fueling"]["active"]
    board_act = st.session_state['activity_state']["Passenger Boarding"]["active"]

    if fuel_act and board_act:
        if not st.session_state['alert_acknowledged']:
            st.session_state['alert_active'] = True
    else:
        # Reset acknowledged state once the danger is gone
        st.session_state['alert_active'] = False
        st.session_state['alert_acknowledged'] = False
    
    # ===========================================================
    # INSIDE THE MAIN LOOP (Updated Trigger)
    # ===========================================================
    # This ensures it runs every 60s as long as ground_power has been seen once
    # Inside Main Loop
    if delay_model and st.session_state['gp_detected_first_time']:
        current_time = time.time()
        if (current_time - st.session_state['last_prediction_time'] > 60):  # Check every 60 seconds
            run_delay_prediction(delay_model)  # Run delay prediction function
            st.session_state['last_prediction_time'] = current_time  # Update last prediction time

# ===========================================================


    # --- DRAW ---
    if f_aoi: draw_transparent_box(annotated, f_aoi, (255, 255, 0), 0.16, "Fueling")
    if c_aoi: draw_transparent_box(annotated, c_aoi, (0, 255, 255), 0.16, st.session_state['baggage_phase'])
    if b_aois:
        draw_transparent_box(annotated, b_aois[0], (0, 255, 0), 0.10, "Pax F")
        draw_transparent_box(annotated, b_aois[1], (0, 255, 0), 0.10, "Pax R")
        
    # ... (Your Logic/Sync to Mongo code) ...

    # 2. Update the Dashboard Panel
    panel = draw_dashboard_panel(720) # Match frame height
    combined = np.hstack((cv2.resize(annotated, (1280, 720)), panel))
    
    # 3. FAST UPDATE
    # Converting to RGB is necessary, but doing it inside the loop is slow. 
    # Ensure this happens right before display.
    image_placeholder.image(
        cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), 
        channels="RGB", 
        use_container_width=True
    )

# ===========================================================
# 6. VIDEO PROCESSING LOOP (Always Outside Tabs)
# ===========================================================

cap.release()