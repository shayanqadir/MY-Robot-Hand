"""
Gesture presets for the robot hand: fist, peace sign, and spread.

Each preset is just a dict of {actuator_name: target_angle_radians}. The
script smoothly interpolates from wherever the hand currently is to the
target preset (rather than snapping), then holds it.

While the sim is running, press:
  1 = fist
  2 = peace sign
  3 = spread (fingers fanned open)
  5 = spider-man sign (middle curled, index + pinky spread, thumb open)
  0 = neutral/open (all joints at 0)

Before running:
  1. pip install mujoco numpy   (in your project's venv)
  2. Update MODEL_PATH below if needed.

These target angles are starting guesses based on your joint layout and
ctrlrange (+/-2.5708 rad, ~147deg per joint) - the hand only has thumb,
index, middle, and pinky fingers (no ring finger in this model), so "peace"
uses index+middle extended with the others curled. Nudge the numbers below
to match how your specific hand geometry actually looks once you see it.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "robothandassembly_fixed.xml"  # update path/filename as needed

TRANSITION_TIME = 0.6  # seconds to smoothly move into a new preset

# --- Gesture presets: actuator_name -> target angle (DEGREES) -----------
# Written in degrees for readability/tuning; converted to radians for MuJoCo
# further down (ctrlrange in the XML is +/-2.5708 rad, i.e. approx +/-147deg).
NEUTRAL_DEG = {
    "Revolute-2_act": -20, "ThumbJoint1_act": 0, "ThumbJoint2_act": 0, "ThumbJoint3_act": 0,
    "Revolute-4_act": 0, "IndexJoint1_act": 0, "IndexJoint2_act": 0, "IndexJoint3_act": 0,
    "Revolute-6_act": 0, "MiddleJoint1_act": 0, "MiddleJoint2_act": 0, "MiddleJoint3_act": 0,
    "Revolute-8_act": 0, "PinkyJoint1_act": 0, "PinkyJoint2_act": 0, "PinkyJoint3_act": 0,
}

FIST_DEG = {
    "Revolute-2_act": 17, "ThumbJoint1_act": 80, "ThumbJoint2_act": -80, "ThumbJoint3_act": 69,
    "Revolute-4_act": 0, "IndexJoint1_act": 80, "IndexJoint2_act": -80, "IndexJoint3_act": 69,
    "Revolute-6_act": 0, "MiddleJoint1_act": -80, "MiddleJoint2_act": -80, "MiddleJoint3_act": 69,
    "Revolute-8_act": 0, "PinkyJoint1_act": 80, "PinkyJoint2_act": -80, "PinkyJoint3_act": 69,
}

PEACE_DEG = {
    "Revolute-2_act": 17, "ThumbJoint1_act": 80, "ThumbJoint2_act": -80, "ThumbJoint3_act": 69,
    "Revolute-4_act": 5, "IndexJoint1_act": 0, "IndexJoint2_act": 0, "IndexJoint3_act": 0,
    "Revolute-6_act": -5, "MiddleJoint1_act": 0, "MiddleJoint2_act": 0, "MiddleJoint3_act": 0,
    "Revolute-8_act": 0, "PinkyJoint1_act": 80, "PinkyJoint2_act": -80, "PinkyJoint3_act": 69,
}

SPREAD_DEG = {
    "Revolute-2_act": 15, "ThumbJoint1_act": 0, "ThumbJoint2_act": 0, "ThumbJoint3_act": 0,
    "Revolute-4_act": 15, "IndexJoint1_act": 0, "IndexJoint2_act": 0, "IndexJoint3_act": 0,
    "Revolute-6_act": 0, "MiddleJoint1_act": 0, "MiddleJoint2_act": 0, "MiddleJoint3_act": 0,
    "Revolute-8_act": -15, "PinkyJoint1_act": 0, "PinkyJoint2_act": 0, "PinkyJoint3_act": 0,
}

# Spider-Man sign: middle finger curled (same joint values as the middle
# finger in FIST), index and pinky extended and fanned outward (abduction
# values borrowed from SPREAD), thumb left open. Tune the abduction angles
# or curl the thumb (copy its values from FIST_DEG) if you want it tighter.
SPIDERMAN_DEG = {
    "Revolute-2_act": -10, "ThumbJoint1_act": 0, "ThumbJoint2_act": 0, "ThumbJoint3_act": 0,
    "Revolute-4_act": 10, "IndexJoint1_act": 0, "IndexJoint2_act": 0, "IndexJoint3_act": 0,
    "Revolute-6_act": 0, "MiddleJoint1_act": -80, "MiddleJoint2_act": -80, "MiddleJoint3_act": 69,
    "Revolute-8_act": -10, "PinkyJoint1_act": 0, "PinkyJoint2_act": 0, "PinkyJoint3_act": 0,
}

# Some joints in this model were built with their hinge axis pointing the
# opposite way (+Y instead of -Y) from the others - checked directly against
# the XML's <joint axis="..."/> vectors. That means a "positive" angle on
# those joints physically curls the opposite direction. -1 here flips it so
# a positive degree value always means the same physical direction (curl/
# flex) on every finger, regardless of that underlying axis quirk.
MIRROR_SIGN = {
    "Revolute-2_act": 1, "ThumbJoint1_act": 1, "ThumbJoint2_act": 1, "ThumbJoint3_act": 1,
    "Revolute-4_act": 1, "IndexJoint1_act": -1, "IndexJoint2_act": -1, "IndexJoint3_act": 1,
    "Revolute-6_act": 1, "MiddleJoint1_act": -1, "MiddleJoint2_act": -1, "MiddleJoint3_act": -1,
    "Revolute-8_act": 1, "PinkyJoint1_act": 1, "PinkyJoint2_act": 1, "PinkyJoint3_act": 1,
}


def _to_radians(preset_deg):
    return {
        name: MIRROR_SIGN.get(name, 1) * np.radians(val)
        for name, val in preset_deg.items()
    }

NEUTRAL = _to_radians(NEUTRAL_DEG)
FIST = _to_radians(FIST_DEG)
PEACE = _to_radians(PEACE_DEG)
SPREAD = _to_radians(SPREAD_DEG)
SPIDERMAN = _to_radians(SPIDERMAN_DEG)

PRESETS = {"0": NEUTRAL, "1": FIST, "2": PEACE, "3": SPREAD, "5": SPIDERMAN}
PRESET_NAMES = {"0": "neutral", "1": "fist", "2": "peace sign", "3": "spread", "5": "spider-man"}


class GestureController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.act_index = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        self.start_ctrl = np.array(data.ctrl, copy=True)
        self.target_ctrl = np.array(data.ctrl, copy=True)
        self.transition_start_time = 0.0
        self.transitioning = False

    def set_preset(self, preset):
        self.start_ctrl = np.array(self.data.ctrl, copy=True)
        self.target_ctrl = np.array(self.data.ctrl, copy=True)
        for name, val in preset.items():
            if name in self.act_index:
                self.target_ctrl[self.act_index[name]] = val
        self.transition_start_time = self.data.time
        self.transitioning = True

    def update(self):
        if not self.transitioning:
            return
        elapsed = self.data.time - self.transition_start_time
        frac = min(elapsed / TRANSITION_TIME, 1.0)
        # smoothstep easing (starts/ends gently rather than linear snapping)
        eased = frac * frac * (3 - 2 * frac)
        self.data.ctrl[:] = self.start_ctrl + eased * (self.target_ctrl - self.start_ctrl)
        if frac >= 1.0:
            self.transitioning = False


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    controller = GestureController(model, data)

    def key_callback(keycode):
        key = chr(keycode) if keycode < 256 else ""
        if key in PRESETS:
            print(f"-> {PRESET_NAMES[key]}")
            controller.set_preset(PRESETS[key])

    print("Press 0=neutral, 1=fist, 2=peace, 3=spread, 5=spider-man")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        start = time.time()
        while viewer.is_running():
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - start
            sleep_time = data.time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
