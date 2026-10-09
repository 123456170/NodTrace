import time
import math
from collections import deque

import numpy as np
import pandas as pd
import cv2
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FPS = 15
FRAME_W, FRAME_H = 640, 480
DEMO_DURATION = 120.0
BUFFER_SEC = 10.0
MEDIAN_SEC = 6.0
WINDOW_SEC = 2.0
STEP_SEC = 0.5

# Stable Face Mesh landmark indices:
# nose tip, chin, right eye outer corner, left eye outer corner,
# right mouth corner, left mouth corner.
FM_INDICES = [1, 152, 33, 263, 61, 291]

# Generic 3D face model, approximately matching FM_INDICES order.
# OpenCV camera convention: x right, y down, z forward.
MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],       # nose tip
        [0.0, 110.0, -15.0],   # chin
        [-55.0, -45.0, 10.0],  # right eye outer corner
        [55.0, -45.0, 10.0],   # left eye outer corner
        [-65.0, 70.0, 0.0],    # right mouth corner
        [65.0, 70.0, 0.0],     # left mouth corner
    ],
    dtype=np.float32,
)

FEATURE_NAMES = [
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "vel_mean",
    "vel_std",
    "vel_min",
    "vel_max",
    "vel_abs_mean",
    "jerk_abs_mean",
    "jerk_abs_max",
    "slow_down_ratio",
    "fast_up_ratio",
    "max_down_rate",
    "max_up_rate",
    "down_before_up",
    "down_up_gap_s",
    "yaw_abs_max",
    "yaw_vel_abs_max",
    "still_ratio",
    "energy",
]


# ----------------------------------------------------------------------------
# Safe conversion helpers
# ----------------------------------------------------------------------------
def _finite_float(x, default=0.0):
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_point(x, y, limit=100000):
    """
    Convert to a Python int tuple that OpenCV can safely parse.
    Returns None if the point is not finite.
    """
    try:
        xf = float(x)
        yf = float(y)

        if not np.isfinite(xf) or not np.isfinite(yf):
            return None

        xi = int(round(xf))
        yi = int(round(yf))

        limit = int(limit)
        xi = max(-limit, min(limit, xi))
        yi = max(-limit, min(limit, yi))

        return (xi, yi)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Geometry / pose helpers
# ----------------------------------------------------------------------------
def get_camera_matrix(w, h):
    focal = max(w, h) * 1.15
    return np.array(
        [
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def euler_to_rotation(pitch_deg, yaw_deg, roll_deg):
    p = math.radians(_finite_float(pitch_deg))
    y = math.radians(_finite_float(yaw_deg))
    r = math.radians(_finite_float(roll_deg))

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def euler_from_rotation(R):
    R = np.asarray(R, dtype=np.float64)

    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return 0.0, 0.0, 0.0

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    pitch = _finite_float(math.degrees(pitch))
    yaw = _finite_float(math.degrees(yaw))
    roll = _finite_float(math.degrees(roll))

    return pitch, yaw, roll


def project_points(points_3d, K):
    points_3d = np.asarray(points_3d, dtype=np.float64)

    if points_3d.ndim == 1:
        if points_3d.size == 3:
            points_3d = points_3d.reshape(1, 3)
        else:
            points_3d = np.zeros((1, 3), dtype=np.float64)

    if points_3d.shape[1] != 3:
        tmp = np.zeros((points_3d.shape[0], 3), dtype=np.float64)
        cols = min(3, points_3d.shape[1])
        if cols > 0:
            tmp[:, :cols] = points_3d[:, :cols]
        points_3d = tmp

    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        K = get_camera_matrix(FRAME_W, FRAME_H)

    if points_3d.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if not np.all(np.isfinite(points_3d)):
        points_3d = np.nan_to_num(points_3d, nan=0.0, posinf=1e6, neginf=-1e6)

    hom = (K @ points_3d.T).T
    z = hom[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    out = hom[:, :2] / z

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def pose_to_landmarks(pitch, yaw, roll, noise=1.0, rng=None, sunglasses=False):
    if rng is None:
        rng = np.random.default_rng()

    R = euler_to_rotation(pitch, yaw, roll)
    tvec = np.array([0.0, 0.0, 500.0], dtype=np.float64)
    pts3 = (R @ MODEL_POINTS.T).T + tvec

    K = get_camera_matrix(FRAME_W, FRAME_H)
    pts2 = project_points(pts3, K)

    pts2 = pts2 + rng.normal(0.0, float(noise), size=pts2.shape)

    if sunglasses:
        # Eyes are indices 2 and 3 in MODEL_POINTS / FM_INDICES order.
        pts2[2:4] = pts2[2:4] + rng.normal(0.0, 2.0, size=(2, 2))

    pts2 = np.nan_to_num(pts2, nan=0.0, posinf=1e6, neginf=-1e6)
    pts2[:, 0] = np.clip(pts2[:, 0], 5.0, FRAME_W - 5.0)
    pts2[:, 1] = np.clip(pts2[:, 1], 5.0, FRAME_H - 5.0)

    return pts2.astype(np.float32)


def estimate_pose_from_landmarks(landmarks, w=FRAME_W, h=FRAME_H):
    if landmarks is None:
        return None

    pts = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)

    if pts.shape[0] < 6:
        return None

    if not np.all(np.isfinite(pts)):
        return None

    pts = pts[:6]
    obj = MODEL_POINTS.astype(np.float64)

    K = get_camera_matrix(w, h).astype(np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            pts,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        return None

    if not ok:
        return None

    try:
        R, _ = cv2.Rodrigues(rvec)
    except Exception:
        return None

    pitch, yaw, roll = euler_from_rotation(R)

    if not np.all(np.isfinite([pitch, yaw, roll])):
        return None

    return float(pitch), float(yaw), float(roll), rvec, tvec, K


def draw_arrowhead(img, pt1, pt2, color, size=12):
    """
    Manual arrowhead to avoid cv2.arrowedLine overload/type problems.
    """
    try:
        x1, y1 = pt1
        x2, y2 = pt2

        dx = float(x2 - x1)
        dy = float(y2 - y1)
        norm = math.hypot(dx, dy)

        if norm < 1.0:
            return

        ux = dx / norm
        uy = dy / norm

        bx = x2 - ux * size
        by = y2 - uy * size

        px = -uy * size * 0.45
        py = ux * size * 0.45

        tri = np.array(
            [
                [x2, y2],
                [bx + px, by + py],
                [bx - px, by - py],
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(img, tri, color, cv2.LINE_AA)
    except Exception:
        pass


def draw_axes(img, rvec, tvec, K, length=70):
    """
    Draw a 3D head-axis overlay.

    Fixed version:
    - does not use cv2.arrowedLine
    - converts all coordinates to safe Python int tuples
    - ignores non-finite projections instead of crashing
    """
    try:
        rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(-1)

        if rvec_arr.size == 3:
            R, _ = cv2.Rodrigues(rvec_arr.reshape(3, 1))
        elif rvec_arr.size == 9:
            R = rvec_arr.reshape(3, 3)
        else:
            return img

        R = np.asarray(R, dtype=np.float64)

        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            return img

        origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        axes = np.array(
            [
                [length, 0.0, 0.0],
                [0.0, length, 0.0],
                [0.0, 0.0, length],
            ],
            dtype=np.float64,
        )

        pts3 = np.vstack([origin, axes])
        proj = project_points(pts3, K)

        if proj.shape[0] < 4:
            return img

        o = _safe_point(proj[0, 0], proj[0, 1])
        if o is None:
            return img

        colors = [
            (0, 0, 255),   # BGR: x axis
            (0, 255, 0),   # BGR: y axis
            (255, 0, 0),   # BGR: z axis
        ]

        for i in range(3):
            p = _safe_point(proj[i + 1, 0], proj[i + 1, 1])
            if p is None or p == o:
                continue

            cv2.line(img, o, p, colors[i], 3, cv2.LINE_AA)
            draw_arrowhead(img, o, p, colors[i], size=12)

        cv2.circle(img, o, 4, (255, 255, 255), -1, cv2.LINE_AA)

    except Exception:
        pass

    return img


# ----------------------------------------------------------------------------
# Ring buffer and signal processing
# ----------------------------------------------------------------------------
class RingBuffer:
    def __init__(self, maxlen):
        self.data = deque(maxlen=int(maxlen))

    def append(self, x):
        self.data.append(x)

    def __len__(self):
        return len(self.data)

    def array(self):
        return np.asarray(self.data, dtype=np.float32)

    def tail(self, n=None):
        arr = self.array()
        if n is None or n >= len(arr):
            return arr
        return arr[-int(n):]


class PoseSignalProcessor:
    """
    Converts raw pose angles into resting-pose-relative signals and computes
    smoothed velocity / jerk. Keeps a 10-second ring buffer.
    """

    def __init__(
        self,
        fps=FPS,
        buffer_sec=BUFFER_SEC,
        median_sec=MEDIAN_SEC,
        alpha=0.45,
        invert_pitch=False,
    ):
        self.fps = float(fps)
        self.buffer_len = max(10, int(fps * buffer_sec))
        self.median_len = max(5, int(fps * median_sec))
        self.alpha = float(alpha)
        self.invert_pitch = bool(invert_pitch)

        self.raw_pitch = RingBuffer(self.buffer_len)
        self.raw_yaw = RingBuffer(self.buffer_len)
        self.raw_roll = RingBuffer(self.buffer_len)
        self.times = RingBuffer(self.buffer_len)

        self.rel_pitch = RingBuffer(self.buffer_len)
        self.rel_yaw = RingBuffer(self.buffer_len)
        self.vel_pitch = RingBuffer(self.buffer_len)
        self.vel_yaw = RingBuffer(self.buffer_len)
        self.jerk_pitch = RingBuffer(self.buffer_len)
        self.jerk_yaw = RingBuffer(self.buffer_len)

        self.prev_rel_p = None
        self.prev_rel_y = None
        self.prev_vel_p = 0.0
        self.prev_vel_y = 0.0
        self.prev_jerk_p = 0.0
        self.prev_jerk_y = 0.0
        self.prev_t = None

    def push(self, pitch, yaw, roll, t):
        pitch = _finite_float(pitch)
        yaw = _finite_float(yaw)
        roll = _finite_float(roll)
        t = _finite_float(t, self.prev_t if self.prev_t is not None else 0.0)

        if self.invert_pitch:
            pitch = -pitch

        self.raw_pitch.append(pitch)
        self.raw_yaw.append(yaw)
        self.raw_roll.append(roll)
        self.times.append(t)

        if len(self.raw_pitch) >= max(5, self.median_len // 3):
            med_p = float(np.median(self.raw_pitch.tail(self.median_len)))
            med_y = float(np.median(self.raw_yaw.tail(self.median_len)))
        else:
            med_p = pitch
            med_y = yaw

        rp = pitch - med_p
        ry = yaw - med_y

        self.rel_pitch.append(rp)
        self.rel_yaw.append(ry)

        if self.prev_t is None or t <= self.prev_t:
            dt = 1.0 / self.fps
        else:
            dt = t - self.prev_t

        if self.prev_rel_p is None:
            vp = 0.0
            vy = 0.0
            jp = 0.0
            jy = 0.0
        else:
            dp = (rp - self.prev_rel_p) / dt
            dy = (ry - self.prev_rel_y) / dt

            vp = self.alpha * dp + (1.0 - self.alpha) * self.prev_vel_p
            vy = self.alpha * dy + (1.0 - self.alpha) * self.prev_vel_y

            jp = (
                self.alpha * ((vp - self.prev_vel_p) / dt)
                + (1.0 - self.alpha) * self.prev_jerk_p
            )
            jy = (
                self.alpha * ((vy - self.prev_vel_y) / dt)
                + (1.0 - self.alpha) * self.prev_jerk_y
            )

        self.vel_pitch.append(vp)
        self.vel_yaw.append(vy)
        self.jerk_pitch.append(jp)
        self.jerk_yaw.append(jy)

        self.prev_rel_p = rp
        self.prev_rel_y = ry
        self.prev_vel_p = vp
        self.prev_vel_y = vy
        self.prev_jerk_p = jp
        self.prev_jerk_y = jy
        self.prev_t = t

        return {
            "t": t,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "rel_pitch": rp,
            "rel_yaw": ry,
            "vel_pitch": vp,
            "vel_yaw": vy,
            "jerk_pitch": jp,
            "jerk_yaw": jy,
        }


# ----------------------------------------------------------------------------
# Rule-based nod detector
# ----------------------------------------------------------------------------
class NodDetector:
    """
    Detects:
      slow downward pitch drift, below max_down deg/s, lasting min_dur..max_dur,
      followed by rapid upward correction, above correction_speed deg/s,
      within correction_window seconds.

    Yaw gate suppresses mirror checks and road-scanning glances.
    """

    def __init__(
        self,
        fps=FPS,
        min_down=2.0,
        max_down=15.0,
        correction_speed=40.0,
        min_dur=0.5,
        max_dur=3.0,
        correction_window=0.4,
        yaw_gate=12.0,
        yaw_vel_gate=18.0,
        cooldown=0.8,
    ):
        self.fps = float(fps)
        self.min_down = float(min_down)
        self.max_down = float(max_down)
        self.correction_speed = float(correction_speed)
        self.min_dur = float(min_dur)
        self.max_dur = float(max_dur)
        self.correction_window = float(correction_window)
        self.yaw_gate = float(yaw_gate)
        self.yaw_vel_gate = float(yaw_vel_gate)
        self.cooldown = float(cooldown)

        self.state = "idle"
        self.events = deque(maxlen=100)
        self.nod_times = deque(maxlen=100)
        self.last_event_t = -100.0

        self.drift_start_t = 0.0
        self.drift_start_p = 0.0
        self.drift_max_p = 0.0
        self.wait_start_t = 0.0
        self.corr_start_t = 0.0
        self.corr_min_vp = 0.0

    def reset(self):
        self.state = "idle"
        self.events.clear()
        self.nod_times.clear()
        self.last_event_t = -100.0

    def _abort(self):
        self.state = "idle"

    def update(self, s):
        t = _finite_float(s.get("t", 0.0))
        p = _finite_float(s.get("rel_pitch", 0.0))
        vp = _finite_float(s.get("vel_pitch", 0.0))
        ry = _finite_float(s.get("rel_yaw", 0.0))
        vy = _finite_float(s.get("vel_yaw", 0.0))

        yaw_ok = (abs(ry) < self.yaw_gate) and (abs(vy) < self.yaw_vel_gate)

        if self.state == "idle":
            if (
                yaw_ok
                and self.min_down < vp < self.max_down
                and (t - self.last_event_t) > self.cooldown
            ):
                self.state = "drift"
                self.drift_start_t = t
                self.drift_start_p = p
                self.drift_max_p = p
            return None

        if self.state == "drift":
            if not yaw_ok:
                self._abort()
                return None

            self.drift_max_p = max(self.drift_max_p, p)

            if vp < -self.correction_speed:
                dur = t - self.drift_start_t
                if self.min_dur <= dur <= self.max_dur:
                    self.state = "correction"
                    self.corr_start_t = t
                    self.corr_min_vp = vp
                else:
                    self._abort()
                return None

            if self.min_down <= vp <= self.max_down:
                return None

            if vp < self.min_down:
                dur = t - self.drift_start_t
                if self.min_dur <= dur <= self.max_dur:
                    self.state = "waiting"
                    self.wait_start_t = t
                else:
                    self._abort()
                return None

            # vp > max_down: too fast downward, not a microsleep drift.
            self._abort()
            return None

        if self.state == "waiting":
            if not yaw_ok:
                self._abort()
                return None

            if (t - self.wait_start_t) > self.correction_window:
                self._abort()
                return None

            if vp < -self.correction_speed:
                self.state = "correction"
                self.corr_start_t = t
                self.corr_min_vp = vp
                return None

            if self.min_down <= vp <= self.max_down:
                if (t - self.drift_start_t) <= self.max_dur:
                    self.state = "drift"
                    self.drift_max_p = max(self.drift_max_p, p)
                    return None

            if (t - self.drift_start_t) > self.max_dur:
                self._abort()

            return None

        if self.state == "correction":
            if not yaw_ok:
                self._abort()
                return None

            self.corr_min_vp = min(self.corr_min_vp, vp)

            correction_over = (
                (vp > -10.0)
                or ((t - self.corr_start_t) > self.correction_window)
            )

            if not correction_over:
                return None

            dur = self.corr_start_t - self.drift_start_t

            if (
                self.min_dur <= dur <= self.max_dur
                and self.corr_min_vp <= -self.correction_speed
            ):
                depth = max(0.0, self.drift_max_p - self.drift_start_p)
                recovery = abs(self.corr_min_vp)

                event = {
                    "t": self.corr_start_t,
                    "start": self.drift_start_t,
                    "end": t,
                    "duration": dur,
                    "depth": float(depth),
                    "recovery_speed": float(recovery),
                }

                self.events.append(event)
                self.nod_times.append(t)
                self.last_event_t = t
                self._abort()
                return event

            self._abort()
            return None

        self._abort()
        return None


# ----------------------------------------------------------------------------
# Analytics helpers
# ----------------------------------------------------------------------------
def alert_active(detector):
    times = list(detector.nod_times)
    if len(times) < 3:
        return False
    return (times[-1] - times[-3]) <= 120.0


def nod_frequency_per_minute(detector, processor, now):
    if now is None or len(detector.nod_times) == 0:
        return 0.0

    arr = processor.times.array()
    first = float(arr[0]) if len(arr) > 0 else float(now) - 60.0
    span = min(60.0, max(10.0, float(now) - first))
    count = sum(1 for t in detector.nod_times if t >= float(now) - span)

    return float(count * (60.0 / span))


def head_still_ratio(processor, threshold=2.0):
    v = processor.vel_pitch.tail(processor.buffer_len)
    y = processor.vel_yaw.tail(processor.buffer_len)

    if len(v) == 0 or len(y) == 0:
        return 1.0

    return float(np.mean((np.abs(v) < threshold) & (np.abs(y) < threshold)))


# ----------------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------------
def generate_demo_timeline(fps=FPS, duration=DEMO_DURATION):
    rng = np.random.default_rng(7)
    t = np.arange(0.0, duration, 1.0 / fps, dtype=np.float64)
    n = len(t)

    # Baseline small natural head motion.
    p = (
        rng.normal(0.0, 0.8, size=n)
        + 1.0 * np.sin(2.0 * np.pi * 0.12 * t + 0.3)
        + rng.normal(0.0, 0.12, size=n)
    )
    y = (
        rng.normal(0.0, 1.2, size=n)
        + 1.8 * np.sin(2.0 * np.pi * 0.08 * t + 1.1)
        + rng.normal(0.0, 0.18, size=n)
    )
    r = (
        rng.normal(0.0, 0.45, size=n)
        + 0.6 * np.sin(2.0 * np.pi * 0.05 * t + 2.0)
        + rng.normal(0.0, 0.08, size=n)
    )

    def add_nod(start, depth, drift_dur, corr_dur):
        m = (t >= start) & (t < start + drift_dur)
        p[m] += depth * (t[m] - start) / drift_dur

        end = start + drift_dur
        m = (t >= end) & (t < end + corr_dur)
        p[m] += depth * (1.0 - (t[m] - end) / corr_dur)

    # Requested demo timeline:
    # 0:50-1:40 -> three clear nods.
    add_nod(55.0, 16.0, 1.60, 0.24)
    add_nod(70.0, 14.0, 1.40, 0.22)
    add_nod(85.0, 18.0, 1.80, 0.26)

    # Extra nod during sunglasses segment to prove robustness.
    add_nod(110.0, 15.0, 1.50, 0.23)

    def add_glance(start, amp, rise=0.40, hold=0.60, fall=0.40):
        t0 = start
        t1 = t0 + rise
        t2 = t1 + hold
        t3 = t2 + fall

        m = (t >= t0) & (t < t1)
        y[m] += amp * (t[m] - t0) / rise

        m = (t >= t1) & (t < t2)
        y[m] += amp

        m = (t >= t2) & (t < t3)
        y[m] += amp * (1.0 - (t[m] - t2) / fall)

    # 0:20-0:50 -> mirror checks / road-scanning glances.
    add_glance(22.0, -28.0, 0.45, 0.70, 0.45)
    add_glance(30.0, 30.0, 0.45, 0.65, 0.45)
    add_glance(38.0, -25.0, 0.40, 0.55, 0.40)
    add_glance(46.0, 27.0, 0.40, 0.55, 0.40)

    sunglasses = (t >= 100.0) & (t < 120.0)
    n_sun = int(np.count_nonzero(sunglasses))

    if n_sun > 0:
        p[sunglasses] += rng.normal(0.0, 0.28, size=n_sun)
        y[sunglasses] += rng.normal(0.0, 0.45, size=n_sun)

    return {
        "t": t,
        "pitch": p,
        "yaw": y,
        "roll": r,
        "sunglasses": sunglasses,
    }


@st.cache_resource
def get_demo_data():
    data = generate_demo_timeline()
    n = len(data["t"])
    landmarks = np.zeros((n, 6, 2), dtype=np.float32)
    rng = np.random.default_rng(11)

    for i in range(n):
        sun = bool(data["sunglasses"][i])
        noise = 2.8 if sun else 1.2

        landmarks[i] = pose_to_landmarks(
            float(data["pitch"][i]),
            float(data["yaw"][i]),
            float(data["roll"][i]),
            noise=noise,
            rng=rng,
            sunglasses=sun,
        )

    data["landmarks"] = landmarks
    return data


def synthesize_subject(sid, duration=45.0, fps=FPS):
    rng = np.random.default_rng(2000 + int(sid))
    t = np.arange(0.0, duration, 1.0 / fps, dtype=np.float64)
    n = len(t)

    p = (
        rng.normal(0.0, 1.2)
        + 1.1 * np.sin(2.0 * np.pi * rng.uniform(0.05, 0.12) * t + rng.uniform(0.0, 2.0 * np.pi))
        + rng.normal(0.0, 0.16, size=n)
    )
    y = (
        rng.normal(0.0, 2.0)
        + 1.8 * np.sin(2.0 * np.pi * rng.uniform(0.04, 0.10) * t + rng.uniform(0.0, 2.0 * np.pi))
        + rng.normal(0.0, 0.22, size=n)
    )
    r = (
        rng.normal(0.0, 0.6)
        + 0.5 * np.sin(2.0 * np.pi * rng.uniform(0.03, 0.08) * t + rng.uniform(0.0, 2.0 * np.pi))
        + rng.normal(0.0, 0.08, size=n)
    )

    true_events = []

    desired = int(rng.integers(3, 6))
    high = max(8.0, duration - 8.0)
    pool = sorted(rng.uniform(5.0, high, size=desired * 3))
    starts = []

    for c in pool:
        if len(starts) >= desired:
            break
        if not starts or (c - starts[-1]) > 7.0:
            starts.append(float(c))

    for start in starts:
        depth = float(rng.uniform(11.0, 20.0))

        drift_dur = float(rng.uniform(1.0, 2.4))
        if depth / drift_dur >= 14.5:
            drift_dur = depth / 13.0
        drift_dur = float(np.clip(drift_dur, 0.7, 2.9))

        corr_dur = float(rng.uniform(0.16, 0.30))
        if depth / corr_dur <= 45.0:
            corr_dur = depth / 60.0
        corr_dur = float(np.clip(corr_dur, 0.14, 0.36))

        m = (t >= start) & (t < start + drift_dur)
        p[m] += depth * (t[m] - start) / drift_dur

        end = start + drift_dur
        m = (t >= end) & (t < end + corr_dur)
        p[m] += depth * (1.0 - (t[m] - end) / corr_dur)

        true_events.append((start, end + corr_dur))

    for _ in range(int(rng.integers(3, 7))):
        start = float(rng.uniform(2.0, max(3.0, duration - 2.0)))
        amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(18.0, 35.0))
        rise = 0.35
        hold = float(rng.uniform(0.30, 0.80))
        fall = 0.35

        t0 = start
        t1 = t0 + rise
        t2 = t1 + hold
        t3 = t2 + fall

        m = (t >= t0) & (t < t1)
        y[m] += amp * (t[m] - t0) / rise

        m = (t >= t1) & (t < t2)
        y[m] += amp

        m = (t >= t2) & (t < t3)
        y[m] += amp * (1.0 - (t[m] - t2) / fall)

    p += rng.normal(0.0, 0.16, size=n)
    y += rng.normal(0.0, 0.22, size=n)

    return {
        "t": t,
        "pitch": p,
        "yaw": y,
        "roll": r,
        "true_events": true_events,
    }


# ----------------------------------------------------------------------------
# Feature extraction and classifier training
# ----------------------------------------------------------------------------
def extract_window_features(t, p, v, j, ry, vy, fps=FPS):
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    ry = np.asarray(ry, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)

    if len(p) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    idx_max = int(np.argmax(v))
    idx_min = int(np.argmin(v))
    gap = float((idx_min - idx_max) / fps) if idx_min > idx_max else -1.0

    feats = {
        "pitch_mean": float(np.mean(p)),
        "pitch_std": float(np.std(p)),
        "pitch_min": float(np.min(p)),
        "pitch_max": float(np.max(p)),
        "pitch_range": float(np.max(p) - np.min(p)),
        "vel_mean": float(np.mean(v)),
        "vel_std": float(np.std(v)),
        "vel_min": float(np.min(v)),
        "vel_max": float(np.max(v)),
        "vel_abs_mean": float(np.mean(np.abs(v))),
        "jerk_abs_mean": float(np.mean(np.abs(j))),
        "jerk_abs_max": float(np.max(np.abs(j))),
        "slow_down_ratio": float(np.mean((v > 2.0) & (v < 15.0))),
        "fast_up_ratio": float(np.mean(v < -40.0)),
        "max_down_rate": float(max(0.0, np.max(v))),
        "max_up_rate": float(max(0.0, -np.min(v))),
        "down_before_up": float(idx_max < idx_min),
        "down_up_gap_s": gap,
        "yaw_abs_max": float(np.max(np.abs(ry))),
        "yaw_vel_abs_max": float(np.max(np.abs(vy))),
        "still_ratio": float(np.mean((np.abs(v) < 2.0) & (np.abs(vy) < 2.0))),
        "energy": float(np.mean(v * v)),
    }

    arr = np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def as_interval(item):
    if isinstance(item, dict):
        return (
            float(item.get("start", item.get("t", 0.0))),
            float(item.get("end", item.get("t", 0.0))),
        )

    try:
        return float(item[0]), float(item[1])
    except Exception:
        return 0.0, 0.0


def center_in_interval(center, intervals, pad=0.2):
    c = float(center)

    for item in intervals:
        a, b = as_interval(item)
        if (a - pad) <= c <= (b + pad):
            return True

    return False


def process_series(data, fps=FPS):
    proc = PoseSignalProcessor(fps=fps)

    out = {
        "t": [],
        "rel_pitch": [],
        "vel_pitch": [],
        "jerk_pitch": [],
        "rel_yaw": [],
        "vel_yaw": [],
    }

    for i in range(len(data["t"])):
        s = proc.push(
            float(data["pitch"][i]),
            float(data["yaw"][i]),
            float(data["roll"][i]),
            float(data["t"][i]),
        )

        out["t"].append(s["t"])
        out["rel_pitch"].append(s["rel_pitch"])
        out["vel_pitch"].append(s["vel_pitch"])
        out["jerk_pitch"].append(s["jerk_pitch"])
        out["rel_yaw"].append(s["rel_yaw"])
        out["vel_yaw"].append(s["vel_yaw"])

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def run_detector_on_arrays(proc, fps=FPS):
    det = NodDetector(fps=fps)
    events = []

    for i in range(len(proc["t"])):
        s = {
            "t": float(proc["t"][i]),
            "rel_pitch": float(proc["rel_pitch"][i]),
            "vel_pitch": float(proc["vel_pitch"][i]),
            "rel_yaw": float(proc["rel_yaw"][i]),
            "vel_yaw": float(proc["vel_yaw"][i]),
        }

        ev = det.update(s)
        if ev is not None:
            events.append(ev)

    return events


def make_windows_from_processed(proc, true_events, detected_events, fps=FPS):
    t = proc["t"]
    p = proc["rel_pitch"]
    v = proc["vel_pitch"]
    j = proc["jerk_pitch"]
    ry = proc["rel_yaw"]
    vy = proc["vel_yaw"]

    win = int(WINDOW_SEC * fps)
    step = max(1, int(STEP_SEC * fps))

    X = []
    y = []
    rule_y = []

    if len(t) < win:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    for start in range(0, len(t) - win + 1, step):
        sl = slice(start, start + win)

        feats = extract_window_features(
            t[sl],
            p[sl],
            v[sl],
            j[sl],
            ry[sl],
            vy[sl],
            fps=fps,
        )

        center = 0.5 * (float(t[start]) + float(t[start + win - 1]))

        label = 1 if center_in_interval(center, true_events, pad=0.20) else 0
        rlabel = 1 if center_in_interval(center, detected_events, pad=0.20) else 0

        X.append(feats)
        y.append(label)
        rule_y.append(rlabel)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=int),
        np.asarray(rule_y, dtype=int),
    )


def make_dataset_for_subject(sid, duration=45.0):
    data = synthesize_subject(sid, duration=duration, fps=FPS)
    proc = process_series(data, fps=FPS)
    detected = run_detector_on_arrays(proc, fps=FPS)

    X, y, rule_y = make_windows_from_processed(
        proc,
        data["true_events"],
        detected,
        fps=FPS,
    )

    return {
        "subject": int(sid),
        "X": X,
        "y": y,
        "rule_y": rule_y,
        "true_events": data["true_events"],
        "detected": detected,
    }


@st.cache_resource
def train_and_evaluate():
    subjects = [0, 1, 2, 3]
    datasets = []

    for sid in subjects:
        ds = make_dataset_for_subject(sid, duration=45.0)
        if len(ds["X"]) > 0 and int(ds["y"].sum()) > 0:
            datasets.append(ds)

    empty_cols = [
        "Subject",
        "Rule Precision",
        "Rule Recall",
        "GBM Precision",
        "GBM Recall",
        "GBM F1",
        "Windows",
        "Positive windows",
    ]

    if not datasets:
        return None, pd.DataFrame(columns=empty_cols), FEATURE_NAMES

    X_all = np.vstack([d["X"] for d in datasets])
    y_all = np.concatenate([d["y"] for d in datasets])
    subj_all = np.concatenate(
        [np.full(len(d["X"]), d["subject"], dtype=int) for d in datasets]
    )

    results = []

    for test_sid in subjects:
        ds = next((d for d in datasets if d["subject"] == test_sid), None)

        if ds is None or len(ds["X"]) == 0:
            continue

        train_mask = subj_all != test_sid
        test_X = ds["X"]
        test_y = ds["y"]
        rule_pred = ds["rule_y"]

        train_pos = int(y_all[train_mask].sum()) if train_mask.any() else 0
        train_neg = int(train_mask.sum() - train_pos) if train_mask.any() else 0

        if train_mask.sum() == 0 or train_pos == 0 or train_neg == 0:
            clf_pred = rule_pred.copy()
        else:
            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.90,
                random_state=42,
            )
            clf.fit(X_all[train_mask], y_all[train_mask])
            clf_pred = clf.predict(test_X)

        results.append(
            {
                "Subject": f"S{test_sid}",
                "Rule Precision": float(
                    precision_score(test_y, rule_pred, zero_division=0)
                ),
                "Rule Recall": float(
                    recall_score(test_y, rule_pred, zero_division=0)
                ),
                "GBM Precision": float(
                    precision_score(test_y, clf_pred, zero_division=0)
                ),
                "GBM Recall": float(
                    recall_score(test_y, clf_pred, zero_division=0)
                ),
                "GBM F1": float(
                    f1_score(test_y, clf_pred, zero_division=0)
                ),
                "Windows": int(len(test_y)),
                "Positive windows": int(test_y.sum()),
            }
        )

    metrics = pd.DataFrame(results, columns=empty_cols)

    final_model = None
    pos = int(y_all.sum())
    neg = int(len(y_all) - pos)

    if pos >= 2 and neg >= 2:
        final_model = GradientBoostingClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.90,
            random_state=42,
        )
        final_model.fit(X_all, y_all)

    return final_model, metrics, FEATURE_NAMES


# ----------------------------------------------------------------------------
# MediaPipe wrapper
# ----------------------------------------------------------------------------
if MP_AVAILABLE:
    class MediaPipeFaceMesh:
        def __init__(self):
            self.face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        def get_landmarks(self, rgb, w, h):
            res = self.face.process(rgb)

            if not res.multi_face_landmarks:
                return None

            lms = res.multi_face_landmarks[0].landmark
            pts = []

            for idx in FM_INDICES:
                pt = lms[idx]
                pts.append([pt.x * float(w), pt.y * float(h)])

            return np.asarray(pts, dtype=np.float32)

        def close(self):
            try:
                self.face.close()
            except Exception:
                pass
else:
    class MediaPipeFaceMesh:
        def __init__(self):
            raise RuntimeError("MediaPipe is not available.")

        def get_landmarks(self, rgb, w, h):
            return None

        def close(self):
            pass


# ----------------------------------------------------------------------------
# Streamlit runtime state
# ----------------------------------------------------------------------------
def reset_runtime():
    invert = False
    if "processor" in st.session_state:
        invert = bool(st.session_state.processor.invert_pitch)

    st.session_state.processor = PoseSignalProcessor(FPS, invert_pitch=invert)
    st.session_state.detector = NodDetector(FPS)
    st.session_state.frame_idx = 0
    st.session_state.history = {"t": [], "rel_pitch": [], "vel_pitch": [], "nod": []}
    st.session_state.classifier_prob = np.nan
    st.session_state.last_s = None
    st.session_state.current_pose = None
    st.session_state.current_frame = None
    st.session_state.plot_fig = None
    st.session_state.webcam_start = None

    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        st.session_state.cap = None

    mp_proc = st.session_state.get("mp_processor")
    if mp_proc is not None:
        try:
            if hasattr(mp_proc, "close"):
                mp_proc.close()
        except Exception:
            pass
        st.session_state.mp_processor = None


def init_state():
    if "demo_data" not in st.session_state:
        st.session_state.demo_data = get_demo_data()

    if "processor" not in st.session_state:
        st.session_state.processor = PoseSignalProcessor(FPS)

    if "detector" not in st.session_state:
        st.session_state.detector = NodDetector(FPS)

    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    if "playing" not in st.session_state:
        st.session_state.playing = True

    if "speed" not in st.session_state:
        st.session_state.speed = 1.0

    if "mode" not in st.session_state:
        st.session_state.mode = "Auto demo"

    if "history" not in st.session_state:
        st.session_state.history = {"t": [], "rel_pitch": [], "vel_pitch": [], "nod": []}

    if "classifier_prob" not in st.session_state:
        st.session_state.classifier_prob = np.nan

    if "last_s" not in st.session_state:
        st.session_state.last_s = None

    if "current_pose" not in st.session_state:
        st.session_state.current_pose = None

    if "current_frame" not in st.session_state:
        st.session_state.current_frame = None

    if "plot_fig" not in st.session_state:
        st.session_state.plot_fig = None

    if "cap" not in st.session_state:
        st.session_state.cap = None

    if "mp_processor" not in st.session_state:
        st.session_state.mp_processor = None

    if "webcam_start" not in st.session_state:
        st.session_state.webcam_start = None


def record_frame(s, ev):
    h = st.session_state.history

    h["t"].append(float(s["t"]))
    h["rel_pitch"].append(float(s["rel_pitch"]))
    h["vel_pitch"].append(float(s["vel_pitch"]))
    h["nod"].append(1 if ev is not None else 0)

    if len(h["t"]) > 3000:
        cut = len(h["t"]) - 3000
        for k in list(h.keys()):
            del h[k][:cut]


def live_classifier_probability():
    model = st.session_state.get("model")

    if model is None:
        return np.nan

    proc = st.session_state.processor
    win = int(WINDOW_SEC * FPS)

    t = proc.times.tail(win)
    p = proc.rel_pitch.tail(win)
    v = proc.vel_pitch.tail(win)
    j = proc.jerk_pitch.tail(win)
    ry = proc.rel_yaw.tail(win)
    vy = proc.vel_yaw.tail(win)

    if len(t) < win:
        return np.nan

    x = extract_window_features(t, p, v, j, ry, vy, fps=FPS).reshape(1, -1)

    try:
        proba = model.predict_proba(x)[0]
        classes = list(model.classes_)

        if 1 in classes:
            return float(proba[classes.index(1)])

        return float(proba[0])
    except Exception:
        return np.nan


# ----------------------------------------------------------------------------
# Demo / webcam frame advancement
# ----------------------------------------------------------------------------
def demo_segment_label(t):
    if t < 20.0:
        return "0:00–0:20 • Head-axis overlay following head motion"
    if t < 50.0:
        return "0:20–0:50 • Mirror checks / road-scanning glances, yaw gate active"
    if t < 100.0:
        return "0:50–1:40 • Slow downward drift + rapid upward correction nods"
    return "1:40–2:00 • Alert window + sunglasses simulation"


def make_synthetic_frame(landmarks, sunglasses, t, rvec=None, tvec=None, K=None):
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    img[:] = (34, 31, 29)

    if landmarks is not None:
        lm = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        pts = []

        for xy in lm:
            sp = _safe_point(float(xy[0]), float(xy[1]))
            if sp is None:
                sp = (FRAME_W // 2, FRAME_H // 2)
            pts.append(sp)

        if len(pts) >= 6:
            pairs = [(0, 1), (2, 3), (4, 5), (2, 4), (3, 5), (0, 2), (0, 3)]

            for a, b in pairs:
                cv2.line(img, pts[a], pts[b], (95, 95, 95), 2, cv2.LINE_AA)

            for pt in pts:
                cv2.circle(img, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

            if sunglasses and len(pts) >= 4:
                p1 = pts[2]
                p2 = pts[3]

                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                norm = math.hypot(dx, dy) + 1e-6

                ux = dx / norm
                uy = dy / norm
                ext = 28.0

                a = _safe_point(p1[0] - ux * ext, p1[1] - uy * ext) or p1
                b = _safe_point(p2[0] + ux * ext, p2[1] + uy * ext) or p2

                cv2.line(img, a, b, (0, 0, 0), 20, cv2.LINE_AA)

                angle = int(math.degrees(math.atan2(uy, ux)))

                mid1 = _safe_point(p1[0] + ux * 10.0, p1[1] + uy * 10.0) or p1
                mid2 = _safe_point(p2[0] - ux * 10.0, p2[1] - uy * 10.0) or p2

                cv2.ellipse(
                    img,
                    mid1,
                    (18, 10),
                    angle,
                    0,
                    360,
                    (0, 0, 0),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.ellipse(
                    img,
                    mid2,
                    (18, 10),
                    angle,
                    0,
                    360,
                    (0, 0, 0),
                    -1,
                    cv2.LINE_AA,
                )

    if rvec is not None and tvec is not None:
        if K is None:
            K = get_camera_matrix(FRAME_W, FRAME_H)
        img = draw_axes(img, rvec, tvec, K, length=70)

    cv2.putText(
        img,
        f"t={float(t):05.1f}s",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    if sunglasses:
        cv2.putText(
            img,
            "SUNGLASSES SIM",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def advance_demo_one_frame():
    data = st.session_state.demo_data
    idx = int(st.session_state.frame_idx)

    if idx >= len(data["t"]):
        st.session_state.playing = False
        return

    lm = data["landmarks"][idx]

    raw_pitch = float(data["pitch"][idx])
    raw_yaw = float(data["yaw"][idx])
    raw_roll = float(data["roll"][idx])
    tt = float(data["t"][idx])

    pose = estimate_pose_from_landmarks(lm, FRAME_W, FRAME_H)
    use_raw = False

    if pose is None:
        use_raw = True
    else:
        pitch, yaw, roll, rvec, tvec, K = pose

        tvec_flat = np.asarray(tvec, dtype=np.float64).reshape(-1)
        if tvec_flat.size >= 3 and float(tvec_flat[2]) <= 1.0:
            use_raw = True

        if abs(float(pitch) - raw_pitch) > 30.0 or abs(float(yaw) - raw_yaw) > 40.0:
            use_raw = True

    if use_raw:
        pitch, yaw, roll = raw_pitch, raw_yaw, raw_roll
        R = euler_to_rotation(pitch, yaw, roll)
        rvec, _ = cv2.Rodrigues(R)
        tvec = np.array([0.0, 0.0, 500.0], dtype=np.float64).reshape(3, 1)
        K = get_camera_matrix(FRAME_W, FRAME_H)
    else:
        pitch, yaw, roll, rvec, tvec, K = pose

    s = st.session_state.processor.push(pitch, yaw, roll, tt)
    ev = st.session_state.detector.update(s)
    st.session_state.classifier_prob = live_classifier_probability()

    img = make_synthetic_frame(
        lm,
        bool(data["sunglasses"][idx]),
        tt,
        rvec=rvec,
        tvec=tvec,
        K=K,
    )

    st.session_state.current_frame = img
    st.session_state.last_s = s
    st.session_state.current_pose = (pitch, yaw, roll, rvec, tvec, K)

    record_frame(s, ev)

    st.session_state.frame_idx = idx + 1


def advance_webcam_one_frame():
    if not MP_AVAILABLE:
        return False

    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    if st.session_state.mp_processor is None:
        try:
            st.session_state.mp_processor = MediaPipeFaceMesh()
        except Exception:
            return False

    cap = st.session_state.cap

    if cap is None or not cap.isOpened():
        return False

    ok, frame = cap.read()
    if not ok:
        return False

    if st.session_state.webcam_start is None:
        st.session_state.webcam_start = time.time()

    elapsed = float(time.time() - float(st.session_state.webcam_start))
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    overlay = frame.copy()

    lms = st.session_state.mp_processor.get_landmarks(rgb, w, h)

    if lms is not None:
        pose = estimate_pose_from_landmarks(lms, w, h)

        if pose is not None:
            pitch, yaw, roll, rvec, tvec, K = pose

            s = st.session_state.processor.push(pitch, yaw, roll, elapsed)
            ev = st.session_state.detector.update(s)
            st.session_state.classifier_prob = live_classifier_probability()

            for pt in lms:
                sp = _safe_point(float(pt[0]), float(pt[1]))
                if sp is not None:
                    cv2.circle(overlay, sp, 4, (0, 255, 255), -1, cv2.LINE_AA)

            overlay = draw_axes(
                overlay,
                rvec,
                tvec,
                K,
                length=max(w, h) * 0.12,
            )

            st.session_state.last_s = s
            record_frame(s, ev)

    cv2.putText(
        overlay,
        f"t={elapsed:05.1f}s",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    st.session_state.current_frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    st.session_state.frame_idx = int(st.session_state.frame_idx) + 1

    return True


# ----------------------------------------------------------------------------
# UI rendering
# ----------------------------------------------------------------------------
def make_plot():
    fig, ax = plt.subplots(figsize=(8.2, 3.1), dpi=110)
    h = st.session_state.history

    if len(h["t"]) < 2:
        ax.text(0.5, 0.5, "Waiting for signal...", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    ts = np.asarray(h["t"], dtype=np.float64)
    ts = ts - ts[0]

    rp = np.asarray(h["rel_pitch"], dtype=np.float64)
    vp = np.asarray(h["vel_pitch"], dtype=np.float64)
    nod = np.asarray(h["nod"], dtype=int)

    max_points = int(FPS * 60)
    if len(ts) > max_points:
        ts = ts[-max_points:]
        rp = rp[-max_points:]
        vp = vp[-max_points:]
        nod = nod[-max_points:]

    ax.plot(ts, rp, color="#1f77b4", lw=1.6, label="Relative pitch")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (deg)")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.18)

    ax2 = ax.twinx()
    ax2.plot(ts, vp, color="#ff7f0e", lw=1.0, alpha=0.75, label="Pitch velocity")
    ax2.set_ylabel("Velocity (deg/s)")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    if np.any(nod > 0):
        idx = np.where(nod > 0)[0]
        ax.scatter(ts[idx], rp[idx], color="red", s=38, zorder=5, label="Detected nod")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


def format_metrics(df):
    if df is None or df.empty:
        return df

    out = df.copy()

    numeric_cols = [
        "Rule Precision",
        "Rule Recall",
        "GBM Precision",
        "GBM Recall",
        "GBM F1",
    ]

    for c in numeric_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{float(x):.2f}")

    for c in ["Windows", "Positive windows"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: str(int(x)))

    return out


def render_sidebar():
    st.sidebar.title("NodTrace controls")

    mode_options = ["Auto demo", "Webcam (if available)"]
    mode_index = 0 if st.session_state.mode == "Auto demo" else 1
    mode = st.sidebar.radio("Source", mode_options, index=mode_index)

    if mode != st.session_state.mode:
        st.session_state.mode = mode
        reset_runtime()

    st.session_state.playing = bool(
        st.sidebar.checkbox("Play / resume", value=bool(st.session_state.playing))
    )

    st.session_state.speed = float(
        st.sidebar.slider(
            "Simulation speed",
            min_value=0.5,
            max_value=4.0,
            value=float(st.session_state.speed),
            step=0.5,
        )
    )

    invert = bool(
        st.sidebar.checkbox(
            "Invert pitch sign",
            value=bool(st.session_state.processor.invert_pitch),
        )
    )

    if invert != bool(st.session_state.processor.invert_pitch):
        st.session_state.processor.invert_pitch = invert
        reset_runtime()

    st.sidebar.button("Reset simulation", on_click=reset_runtime)

    if not MP_AVAILABLE:
        st.sidebar.info(
            "MediaPipe is unavailable in this environment. Auto demo still runs."
        )

    st.sidebar.markdown(
        """
        ### Pipeline
        - 6 Face Mesh landmarks mapped to a generic 3D face model
        - `cv2.solvePnP` for rotation / translation
        - Rolling median removes resting pose
        - Smoothed pitch velocity and pitch jerk
        - 10-second ring buffer
        - Yaw gate suppresses mirror checks / glances
        - Rule detector + gradient-boosted window classifier
        """
    )


def render_ui():
    st.title("NodTrace")
    st.caption(
        "Real-time microsleep nod detection from head-pose time series only."
    )

    s = st.session_state.last_s
    current_time = float(s["t"]) if s is not None else 0.0

    if st.session_state.mode == "Auto demo":
        st.subheader(demo_segment_label(current_time))
    else:
        st.subheader("Live webcam — yaw-gated nod detection")

    col_video, col_metrics = st.columns([2.0, 1.0])

    with col_video:
        if st.session_state.current_frame is not None:
            st.image(st.session_state.current_frame, use_container_width=True)
        else:
            st.info("Preparing live view...")

    with col_metrics:
        detector = st.session_state.detector
        processor = st.session_state.processor

        nod_count = len(detector.events)
        freq = nod_frequency_per_minute(detector, processor, current_time)
        still = head_still_ratio(processor)
        prob = st.session_state.classifier_prob

        st.metric("Nod events", f"{nod_count}")
        st.metric("Nods / min", f"{freq:.1f}")
        st.metric("Head-still ratio", f"{still:.0%}")

        if s is not None:
            st.metric("Relative pitch", f"{float(s['rel_pitch']):.1f}°")
            st.metric("Pitch velocity", f"{float(s['vel_pitch']):.1f}°/s")
            st.metric("Pitch jerk", f"{float(s['jerk_pitch']):.1f}°/s²")
        else:
            st.metric("Relative pitch", "--")
            st.metric("Pitch velocity", "--")
            st.metric("Pitch jerk", "--")

        if prob is None or (isinstance(prob, float) and math.isnan(prob)):
            st.metric("GBM P(nod)", "--")
        else:
            st.metric("GBM P(nod)", f"{float(prob):.0%}")

        latest = detector.events[-1] if len(detector.events) > 0 else None

        if latest is not None:
            st.metric("Last drift depth", f"{float(latest['depth']):.1f}°")
            st.metric("Last recovery speed", f"{float(latest['recovery_speed']):.1f}°/s")

    if alert_active(detector):
        st.error("ALERT: 3 nods within 2 minutes — recommend immediate rest.")
    else:
        st.success("No microsleep alert threshold reached.")

    need_plot = (
        st.session_state.get("plot_fig") is None
        or (not bool(st.session_state.playing))
        or (int(st.session_state.frame_idx) % 4 == 0)
    )

    if need_plot:
        old = st.session_state.get("plot_fig")
        if old is not None:
            try:
                plt.close(old)
            except Exception:
                pass
        st.session_state.plot_fig = make_plot()

    st.pyplot(st.session_state.plot_fig)

    with st.expander(
        "Rule detector vs gradient-boosted classifier, per synthetic subject"
    ):
        metrics_df = st.session_state.get("metrics")

        if metrics_df is None or metrics_df.empty:
            st.write("Evaluation data unavailable.")
        else:
            st.dataframe(format_metrics(metrics_df), use_container_width=True)
            st.caption(
                "Windows are 2.0 s, stepped by 0.5 s. A window is positive when its center "
                "falls inside a true or detected nod interval. Leave-one-subject-out evaluation."
            )


def main():
    st.set_page_config(
        page_title="NodTrace",
        page_icon="😴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_state()

    model, metrics_df, _ = train_and_evaluate()
    st.session_state.model = model
    st.session_state.metrics = metrics_df

    render_sidebar()

    if st.session_state.playing:
        if st.session_state.mode.startswith("Webcam"):
            if not MP_AVAILABLE:
                st.session_state.mode = "Auto demo"
                reset_runtime()
            else:
                ok = advance_webcam_one_frame()
                if not ok:
                    st.session_state.mode = "Auto demo"
                    reset_runtime()

        if st.session_state.mode == "Auto demo":
            if int(st.session_state.frame_idx) >= len(st.session_state.demo_data["t"]):
                reset_runtime()
            advance_demo_one_frame()

        render_ui()

        if st.session_state.playing:
            sleep_s = 1.0 / max(1.0, float(FPS) * float(st.session_state.speed))
            time.sleep(sleep_s)
            st.rerun()
    else:
        if st.session_state.current_frame is None:
            if st.session_state.mode.startswith("Webcam") and MP_AVAILABLE:
                advance_webcam_one_frame()
            else:
                if int(st.session_state.frame_idx) >= len(st.session_state.demo_data["t"]):
                    reset_runtime()
                advance_demo_one_frame()

        render_ui()


if __name__ == "__main__":
    main()