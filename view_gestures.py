"""
Play the hardware gestures in MuJoCo.

    python view_gestures.py                          (macOS: mjpython view_gestures.py)
    python view_gestures.py --xml robothandassembly_fixed.xml

Type a key in the terminal + Enter: 0 1 2 3 4 5 6 8, q = quit.
Uses the same poses, stance, walk and count as send_gestures_to_hardware.py.
Use robothandassembly_crawl.xml (floor + free base) to see the hand actually walk.
"""

import argparse
import queue
import sys
import threading
import time
import types

import numpy as np
import mujoco
import mujoco.viewer

try:                                   # the hardware file imports pyserial but is never opened here
    import serial  # noqa: F401
except ImportError:
    sys.modules["serial"] = types.ModuleType("serial")

import send_gestures_to_hardware as hw
from gesture_presets import MIRROR_SIGN

COUNT_HOLD = 1.0
WALK_LEAD_IN = 3.0   # seconds spent settling into the stance before walking


def make_pose_fn(key):
    presets = {"0": hw.NEUTRAL_DEG, "1": hw.FIST_DEG, "2": hw.PEACE_DEG,
               "3": hw.SPREAD_DEG, "5": hw.SPIDERMAN_DEG}
    if key in presets:
        return lambda t: presets[key]
    if key == "6":
        return lambda t: hw.STANCE_DEG
    if key == "4":
        return lambda t: hw.STANCE_DEG if t < WALK_LEAD_IN else hw.walk_sim_deg(t - WALK_LEAD_IN)
    if key == "8":
        n_up = len(hw.COUNT_ORDER)
        seq = [0] + list(range(1, n_up + 1)) + list(range(n_up - 1, 0, -1)) + [0]

        def count(t):
            i = min(int(t / COUNT_HOLD), len(seq) - 1)
            return hw.count_pose(seq[i])
        return count
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="robothandassembly_fixed.xml")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    act = {model.actuator(i).name: i for i in range(model.nu)}
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    send_hz = hw.GAIT_SEND_HZ
    substeps = max(int(round(1.0 / send_hz / model.opt.timestep)), 1)
    max_step = hw.MAX_SERVO_SPEED_DEG_PER_SEC / send_hz

    # current commanded angle per actuator, in sim degrees (same slew limit as the hardware)
    cur = {n: 0.0 for n in act}
    state = {"fn": make_pose_fn("0"), "t": 0.0}

    cmds = queue.Queue()

    def reader():
        while True:
            try:
                cmds.put(input("key> ").strip())
            except EOFError:
                cmds.put("q")
                return

    threading.Thread(target=reader, daemon=True).start()
    print("Type 0=neutral 1=fist 2=peace 3=spread 4=walk 5=spider-man 6=stance 8=count, q=quit, then Enter.")

    with mujoco.viewer.launch_passive(model, data) as v:
        next_tick = time.perf_counter()
        while v.is_running():
            try:
                cmd = cmds.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd == "q":
                break
            if cmd:
                fn = make_pose_fn(cmd)
                if fn is None:
                    print("unknown key")
                else:
                    state["fn"], state["t"] = fn, 0.0
                    print(f"-> {cmd}")

            pose = state["fn"](state["t"])
            for name, i in act.items():
                target = pose.get(name, 0.0)
                cur[name] += float(np.clip(target - cur[name], -max_step, max_step))
                ctrl = MIRROR_SIGN.get(name, 1) * np.radians(cur[name])
                data.ctrl[i] = np.clip(ctrl, lo[i], hi[i])

            with v.lock():
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
            v.sync()
            state["t"] += 1.0 / send_hz

            next_tick += 1.0 / send_hz
            time.sleep(max(0.0, next_tick - time.perf_counter()))


if __name__ == "__main__":
    main()