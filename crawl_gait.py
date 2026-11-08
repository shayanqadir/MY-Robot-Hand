"""
Open-loop CPG (central pattern generator) crawling gait for the robot hand.

Treats each of the 4 fingers as a "leg" in a trot gait: diagonal pairs move
in phase, the two pairs are 180 degrees out of phase with each other
(Index+Pinky vs Middle+Thumb). Each leg's joints are driven by phase-shifted
sinusoids to approximate a swing-lift-stance-push cycle, the same way simple
quadruped/hexapod gaits are usually bootstrapped before reaching for RL.

Before running this:
  1. Add a ground plane, a <freejoint/> on base_link, and collision geoms on
     the fingertips to the MJCF (see the model-change notes from before).
  2. pip install mujoco --break-system-packages
  3. Update MODEL_PATH below to point at your fixed .xml file.

Tune GAIT_PARAMS below — amplitudes/phases/frequency — to get a working
crawl. This is a starting point, not a finished gait.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "robothandassembly_crawl.xml"  # update path/filename as needed

# Gait timing
FREQUENCY_HZ = 1          # one full stride cycle every 2 seconds
DT = 0.002                   # must match <option timestep> in the XML

# Trot gait: diagonal finger pairs share a phase.
LEG_PHASE = {
    "Index":  0.0,
    "Pinky":  0.0,
    "Middle": np.pi,
    "Thumb":  np.pi,
}

# Per-joint drive: (actuator_name, amplitude_rad, phase_offset_rad, center_rad)
# Joint1 = proximal / main fore-aft swing (largest amplitude)
# Joint2 = lift vs. push (phase-shifted ~90deg so the leg lifts during swing,
#          presses down during stance)
# Joint3 = smaller secondary flexion, roughly follows Joint2
GAIT_PARAMS = []
for finger, abd_act, j1, j2, j3 in [
    ("Index",  "Revolute-4_act", "IndexJoint1_act",  "IndexJoint2_act",  "IndexJoint3_act"),
    ("Middle", "Revolute-6_act", "MiddleJoint1_act", "MiddleJoint2_act", "MiddleJoint3_act"),
    ("Pinky",  "Revolute-8_act", "PinkyJoint1_act",  "PinkyJoint2_act",  "PinkyJoint3_act"),
    ("Thumb",  "Revolute-2_act", "ThumbJoint1_act",  "ThumbJoint2_act",  "ThumbJoint3_act"),
]:
    phase = LEG_PHASE[finger]
    GAIT_PARAMS += [
        (abd_act, 0.05, phase, 0.0),                 # slight abduction sway
        (j1,      0.5,  phase, 0.0),                 # main swing
        (j2,      0.4,  phase + np.pi / 2, 0.3),      # lift/push, offset & flexed bias
        (j3,      0.25, phase + np.pi / 2, 0.2),      # follows j2
    ]


def gait_ctrl(t):
    """Return the full ctrl vector for time t (seconds)."""
    targets = {}
    w = 2 * np.pi * FREQUENCY_HZ
    for name, amp, phase, center in GAIT_PARAMS:
        targets[name] = center + amp * np.sin(w * t + phase)
    return targets


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Map actuator name -> index once, so the per-step loop is cheap.
    act_index = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = data.time
            targets = gait_ctrl(t)
            for name, val in targets.items():
                if name in act_index:
                    data.ctrl[act_index[name]] = val

            mujoco.mj_step(model, data)
            viewer.sync()

            # keep roughly real-time
            elapsed = time.time() - start
            sleep_time = data.time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
