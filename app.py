from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from flask_cors import CORS
import cv2, numpy as np, os, math

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
model = YOLO("yolov8n-pose.pt")

# === Utility functions ===
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def euclid(p1, p2): return float(np.linalg.norm(np.array(p1) - np.array(p2)))
def get_kpt_xy(k, i): return (float(k[i,0]), float(k[i,1]))
def is_visible(k, i, thr=0.25): return float(k[i,2]) >= thr

class RepetitionCounter:
    def __init__(self):
        self.pushup_cnt = 0; self.pushup_state = "up"
        self.squat_cnt = 0; self.squat_state = "top"
        self.jj_cnt = 0; self.jj_state = "closed"

    def update_pushup(self, k):
        if not all(is_visible(k, i) for i in [5,6,7,8,9,10,11,12,15,16]): return
        L = angle(get_kpt_xy(k,5), get_kpt_xy(k,7), get_kpt_xy(k,9))
        R = angle(get_kpt_xy(k,6), get_kpt_xy(k,8), get_kpt_xy(k,10))
        elbow = (L+R)/2
        hipL = angle(get_kpt_xy(k,5), get_kpt_xy(k,11), get_kpt_xy(k,15))
        hipR = angle(get_kpt_xy(k,6), get_kpt_xy(k,12), get_kpt_xy(k,16))
        plank = (hipL > 155 and hipR > 155)
        if plank:
            if self.pushup_state == "up" and elbow < 90:
                self.pushup_state = "down"
            elif self.pushup_state == "down" and elbow > 150:
                self.pushup_cnt += 1; self.pushup_state = "up"

    def update_squat(self, k):
        if not all(is_visible(k, i) for i in [11,12,13,14,15,16]): return
        L = angle(get_kpt_xy(k,11), get_kpt_xy(k,13), get_kpt_xy(k,15))
        R = angle(get_kpt_xy(k,12), get_kpt_xy(k,14), get_kpt_xy(k,16))
        knee = (L+R)/2
        hip_y = (get_kpt_xy(k,11)[1] + get_kpt_xy(k,12)[1]) / 2
        knee_y = (get_kpt_xy(k,13)[1] + get_kpt_xy(k,14)[1]) / 2
        depth_ok = hip_y > knee_y - 5
        if self.squat_state == "top" and knee < 90 and depth_ok:
            self.squat_state = "bottom"
        elif self.squat_state == "bottom" and knee > 160:
            self.squat_cnt += 1; self.squat_state = "top"

    def update_jj(self, k, frame_w):
        if not all(is_visible(k, i) for i in [5,6,9,10,15,16,0]): return
        wristY = (get_kpt_xy(k,9)[1] + get_kpt_xy(k,10)[1]) / 2
        headY  = get_kpt_xy(k,0)[1]
        hands_up = wristY < headY
        ankles = euclid(get_kpt_xy(k,15), get_kpt_xy(k,16))
        feet_apart = ankles > 0.25 * frame_w
        if self.jj_state == "closed" and hands_up and feet_apart:
            self.jj_state = "open"
        elif self.jj_state == "open" and (not hands_up or not feet_apart):
            self.jj_cnt += 1; self.jj_state = "closed"

def draw_text(img, text, x, y, scale=0.9, thickness=2):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

# === API ROUTES ===
@app.route("/api/process", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "Tidak ada file video."}), 400

    file = request.files["video"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    out_path, counter = process_video_yolo(filepath)

    return jsonify({
        "message": "Video berhasil diproses",
        "output_url": f"http://localhost:5000/static/output.mp4",
        "pushup": counter.pushup_cnt,
        "squat": counter.squat_cnt,
        "jj": counter.jj_cnt
    })

@app.route("/api/download", methods=["GET"])
def download_file():
    return send_from_directory(app.config["UPLOAD_FOLDER"], "output.mp4", as_attachment=True)

def process_video_yolo(filepath):
    counter = RepetitionCounter()
    cap = cv2.VideoCapture(filepath)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for r in results:
            if r.keypoints is not None:
                kpts = r.keypoints.data.cpu().numpy()[0]
                counter.update_pushup(kpts)
                counter.update_squat(kpts)
                counter.update_jj(kpts, width)
        draw_text(frame, f"Push-up: {counter.pushup_cnt}", 20, 40)
        draw_text(frame, f"Squat: {counter.squat_cnt}", 20, 80)
        draw_text(frame, f"Jumping Jack: {counter.jj_cnt}", 20, 120)
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path, counter

if __name__ == "__main__":
    app.run(debug=True)
