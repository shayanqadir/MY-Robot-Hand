"""
Send gesture presets to the real hand over USB serial, to the Arduino
running pca9685_servo_bridge.ino, which relays them to the PCA9685 over I2C.

pip install pyserial numpy   (in your project's venv)

Keys: 0=neutral 1=fist 2=peace 3=spread 4=push/pull walk 5=spider-man
      6=stance 8=count  q=quit  (then Enter)

Key 4 walks on the fingertips: the two sides of the palm (thumb+index and
middle+pinky) take turns planting and pulling the hand forward. Ctrl+C stops
the walk and eases back to neutral.

Key 8 makes a fist, counts up 1-4 starting from the thumb, then counts back down.
"""

import sys
import time
import numpy as np
import serial

from gesture_presets import NEUTRAL_DEG, FIST_DEG, PEACE_DEG, SPREAD_DEG, SPIDERMAN_DEG, MIRROR_SIGN

SERIAL_PORT = "COM7"   # Windows: "COM5" etc. | macOS/Linux: "/dev/tty.usbserial-XXXX" or "/dev/ttyUSB0"
BAUD_RATE = 115200

# Which PCA9685 channel (0-15) each actuator's servo is physically wired to.
CHANNEL_ORDER = [
    "Revolute-2_act", "ThumbJoint1_act", "ThumbJoint2_act", "ThumbJoint3_act",
    "Revolute-4_act", "IndexJoint1_act", "IndexJoint2_act", "IndexJoint3_act",
    "Revolute-6_act", "MiddleJoint1_act", "MiddleJoint2_act", "MiddleJoint3_act",
    "Revolute-8_act", "PinkyJoint1_act", "PinkyJoint2_act", "PinkyJoint3_act",
]

SERVO_CENTER_DEG = 90.0

# Per-channel direction override for servos mounted opposite to the sim.
PHYSICAL_TRIM = {
    "IndexJoint1_act": -1,
    "IndexJoint2_act": -1,
    "MiddleJoint2_act": -1,
    "MiddleJoint3_act": -1,
}

# ---------------------------------------------------------------------------
# Walk settings (key 4)
# ---------------------------------------------------------------------------
GAIT_FREQUENCY_HZ = 2           # one full cycle every 2.5 s
GAIT_SEND_HZ = 100  # serial send rate
MAX_SERVO_SPEED_DEG_PER_SEC =500.0
_MAX_STEP_DEG = MAX_SERVO_SPEED_DEG_PER_SEC / GAIT_SEND_HZ

FOOT_ANGLE_DEG = -90.0            # last joint of every finger acts as a foot
RAISE_FOOT_DEG = 0.0              # foot angle while fingers are lifted

SIDE_PHASE = {"Thumb": 0.0, "Index": 0.0, "Middle": 0.5, "Pinky": 0.5}
STANCE_DUTY = 0.6
LIFT_SIGN = 1.0
GAIT_RAMP_SEC = 2.0

STANCE_CURL_J1 = 0.8
STANCE_CURL_J2 = 0.8

THUMB_EXTRA_SPREAD = 110.0
INDEX_EXTRA_SPREAD = 100.0
PINKY_EXTRA_SPREAD = 60.0         # larger = pinky farther from the middle finger
RAISE_CURL = 40.0                 # joints 1+2 curl away from the floor before spreading
STRIDE = 40.0                     # joint1 swing amplitude (sim deg)
LIFT = 50.0                       # joint2 lift amplitude (sim deg)

# ---------------------------------------------------------------------------
# Count settings (key 8)
# ---------------------------------------------------------------------------
COUNT_ORDER = ["Thumb", "Index", "Middle", "Pinky"]
COUNT_PAUSE_SEC = 0.4             # hold time on each number

# (abduction, joint1, joint2, joint3) actuator names per finger
FINGER_ACTS = {
    "Index":  ("Revolute-4_act", "IndexJoint1_act",  "IndexJoint2_act",  "IndexJoint3_act"),
    "Middle": ("Revolute-6_act", "MiddleJoint1_act", "MiddleJoint2_act", "MiddleJoint3_act"),
    "Pinky":  ("Revolute-8_act", "PinkyJoint1_act",  "PinkyJoint2_act",  "PinkyJoint3_act"),
    "Thumb":  ("Revolute-2_act", "ThumbJoint1_act",  "ThumbJoint2_act",  "ThumbJoint3_act"),
}

# Last angle actually sent to each channel, so moves ramp from where the hand is.
_last_sent_deg = {name: SERVO_CENTER_DEG for name in CHANNEL_ORDER}


def _sign(name):
    return MIRROR_SIGN.get(name, 1) * PHYSICAL_TRIM.get(name, 1)


def sim_deg_to_servo_angle(name, sim_deg):
    return float(np.clip(SERVO_CENTER_DEG + _sign(name) * sim_deg, 0.0, 180.0))


# ---------------------------------------------------------------------------
# Stance
# ---------------------------------------------------------------------------
STANCE_DEG = {}


def _build_stance():
    for name in CHANNEL_ORDER:
        STANCE_DEG[name] = SPREAD_DEG.get(name, 0.0)
    for _abd, j1, j2, j3 in FINGER_ACTS.values():
        STANCE_DEG[j1] = SPREAD_DEG.get(j1, 0.0) + STANCE_CURL_J1 * FIST_DEG.get(j1, 0.0)
        STANCE_DEG[j2] = SPREAD_DEG.get(j2, 0.0) + STANCE_CURL_J2 * FIST_DEG.get(j2, 0.0)
        STANCE_DEG[j3] = FOOT_ANGLE_DEG

    def put(finger, deg):
        abd = FINGER_ACTS[finger][0]
        base = SPREAD_DEG.get(abd, 0.0)
        direction = np.sign(base) if base != 0 else 1.0
        STANCE_DEG[abd] = base + direction * deg

    # The middle finger is deliberately left at the preset angle.
    put("Thumb", THUMB_EXTRA_SPREAD)
    put("Index", INDEX_EXTRA_SPREAD)
    put("Pinky", PINKY_EXTRA_SPREAD)


_build_stance()


def _away_from_ground(name):
    """Sim-degree direction that extends a joint (lifts the foot off the ground)."""
    s = np.sign(FIST_DEG.get(name, 0.0))
    return -s if s != 0 else 1.0


def warn_limits(poses, label):
    """Print which joints ask for more than a servo can move (extra is clipped)."""
    worst = {}
    for pose in poses:
        for name in CHANNEL_ORDER:
            raw = SERVO_CENTER_DEG + _sign(name) * pose.get(name, 0.0)
            over = max(0.0 - raw, raw - 180.0, 0.0)
            if over > worst.get(name, 0.0):
                worst[name] = over
    worst = {n: o for n, o in worst.items() if o > 0.5}
    if worst:
        print(f"WARNING ({label}): servo end stop reached, extra degrees are clipped:")
        for name, o in worst.items():
            print(f"   {name}: {o:.0f} deg past the 0/180 limit")


# ---------------------------------------------------------------------------
# Push/pull walk: only joint1 (sweep) and joint2 (lift) move.
# ---------------------------------------------------------------------------
def _smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def walk_sim_deg(t):
    ramp = min(1.0, t / GAIT_RAMP_SEC)
    swing_frac = 1.0 - STANCE_DUTY
    pose = dict(STANCE_DEG)
    for finger, (_abd, j1, j2, _j3) in FINGER_ACTS.items():
        p = (t * GAIT_FREQUENCY_HZ + SIDE_PHASE[finger]) % 1.0
        if p < swing_frac:
            s = p / swing_frac
            x = -1.0 + 2.0 * _smoothstep(s)
            lift = float(np.sin(np.pi * s))
        else:
            x = 1.0 - 2.0 * (p - swing_frac) / STANCE_DUTY
            lift = 0.0
        pose[j1] = STANCE_DEG[j1] + ramp * STRIDE * x
        pose[j2] = STANCE_DEG[j2] + ramp * LIFT_SIGN * _away_from_ground(j2) * LIFT * lift
    return pose


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------
def send_pose(ser, preset_deg):
    angles = [sim_deg_to_servo_angle(name, preset_deg.get(name, 0)) for name in CHANNEL_ORDER]
    for name, a in zip(CHANNEL_ORDER, angles):
        _last_sent_deg[name] = a
    line = ",".join(f"{a:.1f}" for a in angles)
    ser.write((line + "\n").encode())
    return ser.readline().decode().strip()


def _send_slewed(ser, target_sim_deg):
    """Send one frame, moving each servo at most _MAX_STEP_DEG toward target.
    Returns the largest remaining distance (deg) to the target."""
    angles = []
    remaining = 0.0
    for name in CHANNEL_ORDER:
        target = sim_deg_to_servo_angle(name, target_sim_deg.get(name, 0.0))
        current = _last_sent_deg[name]
        new_angle = current + float(np.clip(target - current, -_MAX_STEP_DEG, _MAX_STEP_DEG))
        _last_sent_deg[name] = new_angle
        remaining = max(remaining, abs(target - new_angle))
        angles.append(new_angle)
    line = ",".join(f"{a:.1f}" for a in angles)
    ser.write((line + "\n").encode())
    ser.reset_input_buffer()  # don't block the timing loop waiting for "OK"
    return remaining


def ease_to_pose(ser, pose_deg):
    """Slew-limited move to a pose, with drift-corrected timing."""
    period = 1.0 / GAIT_SEND_HZ
    next_tick = time.perf_counter()
    while _send_slewed(ser, pose_deg) > 0.5:
        next_tick += period
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)


def _current_sim_pose():
    """Where the hand is right now, in sim degrees."""
    pose = {}
    for name in CHANNEL_ORDER:
        s = _sign(name)
        pose[name] = (_last_sent_deg[name] - SERVO_CENTER_DEG) / s if s else 0.0
    return pose


# ---------------------------------------------------------------------------
# Stance and walk
# ---------------------------------------------------------------------------
def go_stance(ser):
    """Lift fingers, spread in the air, then lower into the stance."""
    warn_limits([STANCE_DEG], "stance")
    cur = _current_sim_pose()
    abd_names = [acts[0] for acts in FINGER_ACTS.values()]
    needs_spread = any(
        abs(sim_deg_to_servo_angle(a, STANCE_DEG[a]) - _last_sent_deg[a]) > 5.0
        for a in abd_names
    )
    if needs_spread:
        raised = dict(cur)
        for _abd, j1, j2, j3 in FINGER_ACTS.values():
            raised[j1] = RAISE_CURL * _away_from_ground(j1)
            raised[j2] = RAISE_CURL * _away_from_ground(j2)
            raised[j3] = RAISE_FOOT_DEG
        print("Lifting fingers off the ground...")
        ease_to_pose(ser, raised)
        spread = dict(raised)
        for a in abd_names:
            spread[a] = STANCE_DEG[a]
        print("Spreading in the air...")
        ease_to_pose(ser, spread)
        print("Lowering into the stance...")
    ease_to_pose(ser, STANCE_DEG)


def run_gait(ser):
    """Go to the stance, walk until Ctrl+C, then ease back to neutral."""
    try:
        go_stance(ser)
        warn_limits([walk_sim_deg(GAIT_RAMP_SEC + k / GAIT_SEND_HZ)
                     for k in range(int(GAIT_SEND_HZ / GAIT_FREQUENCY_HZ))], "walk")
        time.sleep(0.5)
        print("Walking... press Ctrl+C to stop.")
        period = 1.0 / GAIT_SEND_HZ
        start = time.perf_counter()
        next_tick = start
        while True:
            _send_slewed(ser, walk_sim_deg(time.perf_counter() - start))
            next_tick += period
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    print("Stopping - easing back to neutral...")
    ease_to_pose(ser, NEUTRAL_DEG)


# ---------------------------------------------------------------------------
# Counting (key 8)
# ---------------------------------------------------------------------------
def count_pose(n):
    """Fist with the first n fingers (in COUNT_ORDER) extended."""
    pose = dict(FIST_DEG)
    for finger in COUNT_ORDER[:n]:
        _abd, j1, j2, j3 = FINGER_ACTS[finger]
        for joint in (j1, j2, j3):
            pose[joint] = NEUTRAL_DEG.get(joint, 0.0)
    return pose


def run_count(ser):
    """Make a fist, count up 1-4 from the thumb, then count back down."""
    try:
        print("Making a fist...")
        ease_to_pose(ser, count_pose(0))
        time.sleep(COUNT_PAUSE_SEC)
        up = list(range(1, len(COUNT_ORDER) + 1))
        down = list(range(len(COUNT_ORDER) - 1, 0, -1))
        for n in up + down:
            print(n)
            ease_to_pose(ser, count_pose(n))
            time.sleep(COUNT_PAUSE_SEC)
        print("Back to a fist.")
        ease_to_pose(ser, count_pose(0))
    except KeyboardInterrupt:
        print("Interrupted - easing back to neutral...")
        ease_to_pose(ser, NEUTRAL_DEG)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PRESETS = {"0": NEUTRAL_DEG, "1": FIST_DEG, "2": PEACE_DEG, "3": SPREAD_DEG, "5": SPIDERMAN_DEG}
PRESET_NAMES = {"0": "neutral", "1": "fist", "2": "peace sign", "3": "spread", "5": "spider-man"}


def main():
    # Windows timers tick every ~15 ms by default; ask for 1 ms so frames arrive evenly.
    timer_fixed = False
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        timer_fixed = True

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Arduino resets when the serial connection opens

    print("Connected. Press 0=neutral 1=fist 2=peace 3=spread 4=walk 5=spider-man 6=stance 8=count, "
          "q=quit, then Enter.")
    try:
        while True:
            key = input("> ").strip()
            if key == "q":
                break
            if key == "4":
                run_gait(ser)
                print("-> walk stopped")
            elif key == "6":
                go_stance(ser)
                print("-> stance (feet down)")
            elif key == "8":
                run_count(ser)
                print("-> count done")
            elif key in PRESETS:
                response = send_pose(ser, PRESETS[key])
                print(f"-> {PRESET_NAMES[key]}: {response}")
            else:
                print("unknown key")
    finally:
        ser.close()
        if timer_fixed:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)


if __name__ == "__main__":
    main()