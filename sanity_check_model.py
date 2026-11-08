"""
Run this BEFORE starting a real training run. Loads robothandassembly_crawl.xml,
steps it with random actions for a few seconds, and reports anything that
would silently sabotage training: NaN/Inf state, the base starting inside
the floor, or immediate instability.

python sanity_check_model.py
"""

import numpy as np
import mujoco

MODEL_PATH = "robothandassembly_crawl.xml"
N_STEPS = 500


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"Loaded model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    assert model.nq == 23, f"Expected 23 qpos (7 freejoint + 16 joints), got {model.nq}"
    assert model.nu == 16, f"Expected 16 actuators, got {model.nu}"

    mujoco.mj_forward(model, data)

    # --- torque + IMU checks ---
    peak = float(np.max(np.abs(model.actuator_forcerange)))
    print(f"Actuator torque ceiling: {peak:.3f} N.m (expected 0.32 = 8x the old 0.04)")
    for nm in ("imu_acc", "imu_gyro", "imu_quat"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, nm)
        assert sid >= 0, f"missing sensor {nm} - is this the IMU-enabled XML?"
    quat = data.sensordata[model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")]:][:4]
    rot = np.empty(9); mujoco.mju_quat2Mat(rot, quat)
    print(f"IMU gravity vector in body frame at rest: {np.round(-rot.reshape(3, 3)[2, :], 3)} "
          f"(should be ~[0, -1, 0]: palm-back = local +Y points up)")

    # --- can every fingertip actually reach the floor? ---
    # The all-finger gait reward needs each tip to be able to touch down. A tip
    # that starts far above the floor and can't get down would just sit under a
    # constant neglect penalty. Heights are the LOWEST point of each tip sphere.
    for nm in ("thumb", "index", "middle", "pinky"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{nm}_tip_grip")
        low = data.geom_xpos[gid][2] - model.geom_size[gid][0]
        print(f"  {nm:>6} tip lowest point at rest: {low * 1000:6.1f} mm above floor")

    z_start = data.qpos[2]
    print(f"Base starting height: {z_start:.4f} m")
    if z_start < 0.01:
        print("WARNING: base starts very close to the floor - check for initial penetration.")

    # Check initial contacts - if fingertip spheres already overlap the
    # floor or each other at rest, that's worth knowing before training.
    if data.ncon > 0:
        print(f"NOTE: {data.ncon} contact(s) present at the initial pose:")
        for i in range(data.ncon):
            c = data.contact[i]
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            print(f"  {geom1} <-> {geom2}")

    print(f"\nStepping {N_STEPS} times with random actions...")
    rng = np.random.default_rng(0)
    ctrl_low = model.actuator_ctrlrange[:, 0]
    ctrl_high = model.actuator_ctrlrange[:, 1]

    for step in range(N_STEPS):
        data.ctrl[:] = rng.uniform(ctrl_low, ctrl_high)
        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            print(f"FAIL at step {step}: NaN/Inf in state. Likely an unstable "
                  f"contact or joint - check damping/armature and collision geom sizes.")
            return

        z = data.qpos[2]
        if z < -0.5 or z > 5.0:
            print(f"FAIL at step {step}: base height blew up to {z:.2f} m - "
                  f"likely an exploding contact force.")
            return

    print(f"\nOK: ran {N_STEPS} steps with random actions, no NaN/Inf, "
          f"final base height={data.qpos[2]:.4f} m. Model looks stable enough to train.")


if __name__ == "__main__":
    main()
