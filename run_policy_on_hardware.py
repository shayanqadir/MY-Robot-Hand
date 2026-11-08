"""
Drive the real hand with the TRAINED PPO POLICY (not the hand-tuned CPG
gait) over the same USB serial link send_gestures_to_hardware.py uses.

WHY THIS SCRIPT STILL SHADOWS THE SIM INSTEAD OF CLOSING THE LOOP ON HARDWARE:
hand_env.py's observation is now IMU-style: gyro, body-frame gravity
direction, the last action sent, and a gait clock. That is all hardware-
realisable, but only once the real hand actually streams IMU data back to
this PC (the current Arduino bridge only receives servo angles). Until that
return path exists, this script feeds the policy the SIM's IMU readings, not
the real hand's. Closing the loop = replace the gyro/gravity part of `obs`
with real IMU values, expressed in the same axes as the 'imu' site in the
MJCF (see its comment).

What this script does instead: keeps stepping HandCrawlEnv exactly like
watch_policy.py (so the policy always gets the observation it was trained
on), and on every control step also mirrors the resulting action out to
the real servos. The real hand reproduces the learned gait open-loop; it
will NOT self-correct if it physically drifts from what the sim is doing,
so watch it the first few cycles and be ready to hit 'q'/kill power.

pip install stable-baselines3 gymnasium mujoco numpy pyserial   (in your venv)

Run: python run_policy_on_hardware.py hand_crawl_ppo_final.zip
  or: python run_policy_on_hardware.py checkpoints/hand_crawl_ppo_500000_steps.zip
"""

import os
import sys
import time
import numpy as np
import serial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from hand_env import HandCrawlEnv
from send_gestures_to_hardware import (
    SERIAL_PORT, BAUD_RATE, CHANNEL_ORDER, sim_deg_to_servo_angle,
)

VECNORMALIZE_PATH = "vecnormalize_stats.pkl"

# Serial + servos can't keep up with the sim's 100Hz control rate (that's
# WHY send_gestures_to_hardware.py's CPG gait only sends at 30Hz and skips
# waiting for the "OK" reply). Same reasoning here: step the sim at its
# native 100Hz for policy accuracy, but only forward every Nth action to
# hardware.
HARDWARE_SEND_HZ = 25
SIM_CONTROL_HZ = 100  # 1 / (opt.timestep * FRAME_SKIP) = 1/(0.002*5)  (was wrongly 20 -> sent EVERY step, 4x too fast)

# SLEW-RATE LIMIT (the fix for "movement is too fast/sudden and will break
# the motor"): the raw policy action can swing a large number of degrees
# from one hardware send to the next - nothing during training penalized
# how fast the TARGET changes, only how much torque/effort it took to hit
# it, so there was never pressure to keep setpoints smooth. Unlike
# gesture_presets.py's GestureController (which eases between two known
# poses over TRANSITION_TIME), this loop has no "next pose", just a stream
# of raw targets - so instead we cap how many degrees any single joint is
# allowed to move PER SEND, and step gradually toward the real target
# rather than jumping straight to it. Tune this down further if the motor
# still strains, or up if the gait starts looking too sluggish/lagged
# relative to what's happening in the sim viewer.
MAX_SERVO_SPEED_DEG_PER_SEC = 60.0
_MAX_STEP_DEG = MAX_SERVO_SPEED_DEG_PER_SEC / HARDWARE_SEND_HZ

# Ramp state: last angle actually sent to each channel, so we always know
# how far we're allowed to move this send. Starts at the servo center (90)
# rather than the policy's first raw target, so the very first send also
# ramps up smoothly instead of snapping from wherever the hand physically
# is right now.
_last_sent_deg = {name: 90.0 for name in CHANNEL_ORDER}


def send_action_to_hardware(ser, action_rad):
    """action_rad is env.step()'s action array, in MODEL actuator order.
    That order is the same as CHANNEL_ORDER (verified against the MJCF -
    both list Revolute-2, Thumb1-3, Revolute-4, Index1-3, ... in the same
    sequence), so this is a direct positional zip, no remapping needed."""
    sim_deg = {
        name: float(np.degrees(a))
        for name, a in zip(CHANNEL_ORDER, action_rad)
    }
    target_servo_deg = {
        name: sim_deg_to_servo_angle(name, sim_deg[name]) for name in CHANNEL_ORDER
    }

    angles = []
    for name in CHANNEL_ORDER:
        current = _last_sent_deg[name]
        target = target_servo_deg[name]
        delta = np.clip(target - current, -_MAX_STEP_DEG, _MAX_STEP_DEG)
        new_angle = current + delta
        _last_sent_deg[name] = new_angle
        angles.append(new_angle)

    line = ",".join(f"{a:.1f}" for a in angles)
    ser.write((line + "\n").encode())
    ser.reset_input_buffer()  # don't block the timing loop waiting for "OK"


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_policy_on_hardware.py <path_to_model.zip>")
        sys.exit(1)

    model_path = sys.argv[1]
    policy = PPO.load(model_path)

    env = HandCrawlEnv(render_mode="human", randomize=False)

    vecnorm = None
    if os.path.exists(VECNORMALIZE_PATH):
        dummy = DummyVecEnv([lambda: HandCrawlEnv()])
        vecnorm = VecNormalize.load(VECNORMALIZE_PATH, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
        print(f"Loaded observation normalization from {VECNORMALIZE_PATH}")
    else:
        print("No vecnormalize_stats.pkl found - proceeding without normalization "
              "(only correct if you trained without VecNormalize).")

    print(f"Connecting to {SERIAL_PORT} @ {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Arduino resets when the serial connection opens

    input("Hand should be resting clear of obstructions. Press Enter to start "
          "streaming the policy to hardware (Ctrl+C to stop)...")

    obs, _ = env.reset()
    send_every = max(round(SIM_CONTROL_HZ / HARDWARE_SEND_HZ), 1)
    step_count = 0

    try:
        while True:
            obs_in = vecnorm.normalize_obs(obs) if vecnorm else obs
            action, _ = policy.predict(obs_in, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if step_count % send_every == 0:
                send_action_to_hardware(ser, action)

            step_count += 1
            time.sleep(env.model.opt.timestep * 5)  # keep pace with FRAME_SKIP, ~real-time

            if terminated or truncated:
                print(f"Sim episode ended (x_position={info['x_position']:.3f}). "
                      "Resetting sim - re-seat the real hand to a similar neutral "
                      "pose now if it has drifted, then Enter to continue.")
                input()
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()
        ser.close()


if __name__ == "__main__":
    main()
