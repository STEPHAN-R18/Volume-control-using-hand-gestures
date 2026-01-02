from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol, _ = volume.GetVolumeRange()

current_vol = 0
prev_vol = 0
current_distance = 0
prev_distance = 0
mute_status = False
last_stable_time = time.time()
pinch_start = None

SMOOTHING = 0.2
STABLE_THRESHOLD = 5
PINCH_DISTANCE = 25
PINCH_TIME = 2

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_volume')
def get_volume():
    return jsonify({
        "vol": 0 if mute_status else current_vol,
        "dist": int(current_distance),
        "mute": mute_status
    })

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global current_vol, prev_vol, current_distance
    global mute_status, last_stable_time, pinch_start, prev_distance

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))

                if lmList:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]

                    dist = math.hypot(x2 - x1, y2 - y1)
                    current_distance = dist

                    # ---- PINCH MUTE ----
                    if dist < PINCH_DISTANCE:
                        if pinch_start is None:
                            pinch_start = time.time()
                        elif time.time() - pinch_start >= PINCH_TIME:
                            mute_status = not mute_status
                            pinch_start = None
                    else:
                        pinch_start = None

                    if abs(dist - prev_distance) < STABLE_THRESHOLD:
                        if time.time() - last_stable_time > 0.3:
                            target_vol = np.interp(dist, [20, 200], [0, 100])
                            current_vol = int(prev_vol + (target_vol - prev_vol) * SMOOTHING)
                            prev_vol = current_vol

                            if not mute_status:
                                vol = np.interp(current_vol, [0, 100], [minVol, maxVol])
                                volume.SetMasterVolumeLevel(vol, None)
                    else:
                        last_stable_time = time.time()

                    prev_distance = dist

                    cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # ---- STATUS ----
        status_text = "MUTED" if mute_status else "ACTIVE"
        status_color = (0, 0, 255) if mute_status else (0, 255, 0)
        cv2.putText(img, f"Status: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # ---- VOLUME BAR ----
        bar_x1, bar_x2 = 150, 500
        bar_y1, bar_y2 = h - 40, h - 20
        fill_x = int(np.interp(current_vol, [0, 100], [bar_x1, bar_x2]))

        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (255,255,255), 2)
        cv2.rectangle(img, (bar_x1, bar_y1), (fill_x, bar_y2), (0,255,0), cv2.FILLED)
        cv2.putText(img, f"{current_vol}%", (bar_x2 + 10, bar_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
