"""
Gymnasium environment: the robot hand, learning to crawl, with domain
randomization (friction, mass, servo strength, action latency, observation
noise, random shoves).

Requires robothandassembly_crawl.xml (ground plane, freejoint on base_link,
fingertip collision geoms, 'floor' and 'base_link_collision' geoms, and the
'imu' site + imu_gyro / imu_quat sensors).

OBSERVATION (hardware-realisable, 24-dim) - only things the real hand can know:
  gyro (3)            body-frame angular velocity from the IMU
  gravity (3)         gravity direction in the body frame, from the IMU's
                      orientation quaternion (upright palm-up -> constant vector)
  last action (16)   the joint targets sent last step (servos have no feedback,
                      so the commanded angles are the best joint-state estimate)
  clock sin/cos (2)   a fixed-frequency phase signal so the policy has a gait
                      timebase without joint-velocity / base-velocity feedback
Base height, base linear velocity and joint velocities are NOT observed
(no sensor for them on the real hand). Set USE_FULL_STATE_OBS = True to append
the old privileged state (qpos[2:] + qvel) for a sim-only fallback.

pip install gymnasium mujoco numpy

Model DOF layout (fixed by the XML):
  qpos = [x, y, z, qw, qx, qy, qz,  <16 joint angles>]   (23 total)
  qvel = [vx, vy, vz, wx, wy, wz,   <16 joint velocities>] (22 total)
  ctrl = <16 position-actuator targets, radians>            (16 total)

RANDOMIZATION
  train_ppo.py calls env.env_method("set_randomization_scale", s) with s in
  [0, 1]. Every range below is multiplied by s, so s=0 is the clean task and
  s=1 is the full range. The new scale takes effect at the next reset().
  The scale starts at 0.0, so watch_policy.py (which builds a plain
  HandCrawlEnv()) and run_policy_on_hardware.py (randomize=False) always run
  the clean environment.
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

MODEL_PATH = "robothandassembly_crawl.xml"

FRAME_SKIP = 5          # physics substeps per env step -> control runs at 1/(0.002*5) = 100 Hz
MAX_EPISODE_STEPS = 1000  # 10 seconds of sim time per episode at 100 Hz

# Reward weights - the usual first knobs to turn if training behaves oddly
FORWARD_REWARD_WEIGHT = 30.0  # was 5.0. At ~0.05 m/s a crawl paid only 0.25/step while
                              # just standing still paid ~4.5/step, so PPO had little reason to move.
ALIVE_BONUS = 0.5
CONTROL_COST_WEIGHT = 0.01
UPRIGHT_REWARD_WEIGHT = 0.5
HEADING_REWARD_WEIGHT = 0.5   # keeps the hand pointed along +X instead of yawing away

# Without these two, nothing in the reward discourages resting the palm
# flat on the floor - it's the cheapest way to stay upright/stable, so
# that's exactly what PPO finds. PALM_CONTACT_PENALTY makes that touch
# directly costly; PALM_HEIGHT_WEIGHT additionally rewards actually hovering
# at TARGET_PALM_HEIGHT so there's a positive incentive, not just an
# avoidance signal.
PALM_CONTACT_PENALTY_WEIGHT = 3.0
PALM_HEIGHT_WEIGHT = 1.0   # was 3.0 - a big 'hover here' bonus rewards standing still
TARGET_PALM_HEIGHT = 0.06     # meters above the floor
PALM_HEIGHT_TOLERANCE = 0.015  # full reward within +/- this band around target

# --- All-finger gait shaping -------------------------------------------
# The four fingertips ('thumb/index/middle/pinky_tip_grip' geoms) are the feet.
ACTION_RATE_PENALTY_WEIGHT = 0.1   # was 0.02 (negligible next to ~2.5/step alive rewards)
ACTION_SMOOTHING = 0.35   # low-pass on commands: ctrl += alpha*(action-ctrl). ~30 ms time constant
                          # at 100 Hz. Turns per-step random noise into smooth joint motion.
LATERAL_PENALTY_WEIGHT = 5.0   # cost per m/s of sideways (y) drift
AIR_TIME_MIN = 0.10      # s: a tip must be off the floor at least this long before a touchdown pays
AIR_TIME_CAP = 0.40      # s: ...and the payout stops growing past this
AIR_TIME_WEIGHT = 2.0    # stepping bonus, only paid while the hand is making forward progress
NEGLECT_AFTER_S = 1.0    # a finger that hasn't touched the floor for this long starts costing reward
NEGLECT_RAMP_S = 1.0     # ...ramping to the full penalty over this many more seconds
NEGLECT_PENALTY_WEIGHT = 0.3   # (max total over all fingers) - forces all four fingers to take part

# Episode ends early if the base falls outside this height band (m)
MIN_HEALTHY_Z = 0.015
MAX_HEALTHY_Z = 0.15

# ---------------------------------------------------------------------------
# Domain randomization ranges at scale = 1.0 (all multiplied by the scale)
# ---------------------------------------------------------------------------
FRICTION_RANGE = 0.30        # sliding friction multiplied by U(1-r, 1+r), one draw per episode
MASS_RANGE = 0.20            # per-body mass (and inertia) multiplied by U(1-r, 1+r)
SERVO_STRENGTH_RANGE = 0.25  # per-actuator kp multiplied by U(1-r, 1+r)

MAX_LATENCY_STEPS = 2        # action delay of 0..2 control steps (0-20 ms at 100 Hz)

CLOCK_HZ = 2.0               # gait-clock frequency fed to the policy (sin/cos of 2*pi*f*t)
IMU_NOISE_GYRO = 0.05        # rad/s std of gyro noise at full scale
IMU_NOISE_GRAV = 0.02        # std of noise on the body-frame gravity direction at full scale
USE_FULL_STATE_OBS = False   # True = also append privileged qpos[2:] + qvel (sim-only, not deployable)
OBS_NOISE_POS = 0.01         # (full-state mode only) std of gaussian noise on the qpos part
OBS_NOISE_VEL = 0.10         # (full-state mode only) std of gaussian noise on the qvel part

SHOVE_PROB_PER_STEP = 0.005  # ~ one shove per 2 s of sim time at full scale
SHOVE_ACCEL_MAX = 3.0        # m/s^2 of push on the base at full scale (force = mass * accel)
SHOVE_DURATION_STEPS = 3     # control steps a shove lasts (30 ms)


class HandCrawlEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, randomize=True):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.randomize = randomize
        self._viewer = None
        self._renderer = None

        self.nu = self.model.nu  # 16 actuators
        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(low=ctrl_low.astype(np.float32),
                                        high=ctrl_high.astype(np.float32),
                                        dtype=np.float32)

        # obs = [gyro(3), gravity_body(3), last_action(nu), clock sin/cos(2)] (+ full state, optional)
        self._dt = self.model.opt.timestep * FRAME_SKIP
        self._imu_gyro_adr = self._sensor_adr("imu_gyro", 3)
        self._imu_quat_adr = self._sensor_adr("imu_quat", 4)
        self._n_full_state = (self.model.nq - 2) + self.model.nv
        obs_dim = 6 + self.nu + 2 + (self._n_full_state if USE_FULL_STATE_OBS else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(obs_dim,), dtype=np.float32)

        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self._base_id < 0:
            raise RuntimeError("No body named 'base_link' in the MJCF.")

        self._floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._palm_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "base_link_collision")
        if self._floor_gid < 0 or self._palm_gid < 0:
            raise RuntimeError(
                "Expected geoms 'floor' and 'base_link_collision' in the MJCF "
                "for the palm-contact penalty - check names haven't changed."
            )

        # Fingertip geoms = the four "feet" used for the gait shaping.
        tip_names = ["thumb_tip_grip", "index_tip_grip", "middle_tip_grip", "pinky_tip_grip"]
        tip_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in tip_names]
        if min(tip_ids) < 0:
            raise RuntimeError(f"Missing fingertip geom(s) in the MJCF: {tip_names}")
        self._tip_gids = np.array(tip_ids, dtype=np.int64)
        self._tips_prev = np.zeros(len(tip_ids), dtype=bool)
        self._air_time = np.zeros(len(tip_ids))
        self._since_contact = np.zeros(len(tip_ids))
        self._last_action = np.zeros(self.nu)
        self._smoothed = np.zeros(self.nu)   # low-passed command actually sent

        self._init_qpos = self.data.qpos.copy()
        self._elapsed_steps = 0

        # --- domain randomization state ---
        # Nominal physics, cached once so every reset() randomizes around the
        # ORIGINAL values instead of compounding on the previous episode's.
        self._nom_friction = self.model.geom_friction.copy()
        self._nom_mass = self.model.body_mass.copy()
        self._nom_inertia = self.model.body_inertia.copy()
        self._nom_kp_gain = self.model.actuator_gainprm[:, 0].copy()
        self._nom_kp_bias = self.model.actuator_biasprm[:, 1].copy()
        self._total_mass = float(np.sum(self._nom_mass))

        self._rand_scale = 0.0   # set by set_randomization_scale(); 0 = clean env
        self._latency = 0
        self._action_buffer = deque(maxlen=MAX_LATENCY_STEPS + 1)
        self._shove_steps_left = 0

        noise = [np.full(3, IMU_NOISE_GYRO), np.full(3, IMU_NOISE_GRAV),
                 np.zeros(self.nu),   # last action is exact (we sent it)
                 np.zeros(2)]         # clock is exact
        if USE_FULL_STATE_OBS:
            noise += [np.full(self.model.nq - 2, OBS_NOISE_POS), np.full(self.model.nv, OBS_NOISE_VEL)]
        self._obs_noise_std = np.concatenate(noise).astype(np.float64)

    def _sensor_adr(self, name, dim):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"No sensor named '{name}' in the MJCF - use the IMU-enabled "
                               f"robothandassembly_crawl.xml.")
        assert self.model.sensor_dim[sid] == dim, (name, self.model.sensor_dim[sid])
        return int(self.model.sensor_adr[sid])

    # ------------------------------------------------------------------
    # Randomization
    # ------------------------------------------------------------------
    def set_randomization_scale(self, scale):
        """Called by the training curriculum via env_method. Clamped to
        [0, 1]. Takes effect at the next reset()."""
        self._rand_scale = float(np.clip(scale, 0.0, 1.0))

    def _randomize_physics(self):
        """Resample friction / mass / servo strength / latency for this
        episode. Always writes from the cached nominal values, so scale 0
        (or randomize=False) restores the clean model exactly."""
        s = self._rand_scale if self.randomize else 0.0
        rng = self.np_random
        m = self.model

        if s > 0.0:
            fric = rng.uniform(1 - FRICTION_RANGE * s, 1 + FRICTION_RANGE * s)
            mass = rng.uniform(1 - MASS_RANGE * s, 1 + MASS_RANGE * s,
                               size=self._nom_mass.shape)
            kp = rng.uniform(1 - SERVO_STRENGTH_RANGE * s, 1 + SERVO_STRENGTH_RANGE * s,
                             size=self._nom_kp_gain.shape)
            self._latency = int(rng.integers(0, int(round(MAX_LATENCY_STEPS * s)) + 1))
        else:
            fric = 1.0
            mass = np.ones_like(self._nom_mass)
            kp = np.ones_like(self._nom_kp_gain)
            self._latency = 0

        # Sliding friction only (column 0); torsional/rolling stay nominal.
        m.geom_friction[:] = self._nom_friction
        m.geom_friction[:, 0] = self._nom_friction[:, 0] * fric

        # Scale inertia by the same factor as mass so the two stay consistent.
        m.body_mass[:] = self._nom_mass * mass
        m.body_inertia[:] = self._nom_inertia * mass[:, None]

        # Position actuators: gainprm[0] = kp, biasprm[1] = -kp. Scale both
        # by the same factor or you get a steady-state offset.
        m.actuator_gainprm[:, 0] = self._nom_kp_gain * kp
        m.actuator_biasprm[:, 1] = self._nom_kp_bias * kp

        # Refresh derived constants (e.g. subtree masses) after editing mass.
        mujoco.mj_setConst(m, self.data)

    def _update_shove(self):
        """Advance / start a random horizontal push on the base."""
        base = self._base_id
        if self._shove_steps_left > 0:
            self._shove_steps_left -= 1
            if self._shove_steps_left == 0:
                self.data.xfrc_applied[base, :3] = 0.0
            return

        s = self._rand_scale if self.randomize else 0.0
        if s > 0.0 and self.np_random.random() < SHOVE_PROB_PER_STEP * s:
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
            accel = self.np_random.uniform(0.5, 1.0) * SHOVE_ACCEL_MAX * s
            force = accel * self._total_mass
            self.data.xfrc_applied[base, 0] = force * np.cos(angle)
            self.data.xfrc_applied[base, 1] = force * np.sin(angle)
            self.data.xfrc_applied[base, 2] = 0.0
            self._shove_steps_left = SHOVE_DURATION_STEPS

    # ------------------------------------------------------------------
    # Observation / helpers
    # ------------------------------------------------------------------
    def _get_obs(self):
        d = self.data
        gyro = d.sensordata[self._imu_gyro_adr:self._imu_gyro_adr + 3]
        quat = d.sensordata[self._imu_quat_adr:self._imu_quat_adr + 4]
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, quat)                 # IMU frame -> world
        gravity_body = -rot.reshape(3, 3)[2, :]        # R^T @ [0, 0, -1]
        phase = 2.0 * np.pi * CLOCK_HZ * self._elapsed_steps * self._dt
        parts = [gyro, gravity_body, self._smoothed, [np.sin(phase), np.cos(phase)]]
        if USE_FULL_STATE_OBS:
            parts += [d.qpos[2:], d.qvel]
        obs = np.concatenate(parts)
        s = self._rand_scale if self.randomize else 0.0
        if s > 0.0:
            obs = obs + self.np_random.normal(0.0, 1.0, size=obs.shape) * self._obs_noise_std * s
        return obs.astype(np.float32)

    def _is_healthy(self):
        z = self.data.qpos[2]
        upright = self.data.xmat[self._base_id].reshape(3, 3)[2, 1]
        return (MIN_HEALTHY_Z <= z <= MAX_HEALTHY_Z
                and upright > 0.0            # ends the episode once it rolls past 90 deg
                and np.all(np.isfinite(self.data.qpos)))

    def _base_rotation(self):
        """3x3 world rotation matrix of base_link."""
        return self.data.xmat[self._base_id].reshape(3, 3)

    def _upright_reward(self):
        """+1 when the palm-back faces straight up, -1 when flipped over.

        In base_link's CAD frame the palm-back normal is local +Y (see the
        euler comment in the MJCF). R[2, 1] is the world-z component of
        that local +Y axis, so it's +1 upright and -1 inverted. Using the
        rotation matrix avoids the quaternion double-cover sign trap (an
        earlier version read qpos[6], which is qz/yaw, and rewarded
        spinning around to face backwards).
        """
        return self._base_rotation()[2, 1]

    def _heading_reward(self):
        """+1 when the fingers point along world +X, the reward direction.

        The fingers' length axis is base_link local -Z, so world forward is
        -R[:, 2] and its x-component is -R[0, 2].
        """
        return -self._base_rotation()[0, 2]

    def _floor_contacts(self):
        """One pass over data.contact -> (palm_touching_floor, tip_touching_floor[4]).
        Tips are ordered thumb, index, middle, pinky."""
        palm = False
        tips = np.zeros(len(self._tip_gids), dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self._floor_gid:
                other = c.geom2
            elif c.geom2 == self._floor_gid:
                other = c.geom1
            else:
                continue
            if other == self._palm_gid:
                palm = True
            else:
                hit = np.nonzero(self._tip_gids == other)[0]
                if hit.size:
                    tips[hit[0]] = True
        return palm, tips

    def _height_reward(self):
        """1.0 at TARGET_PALM_HEIGHT, falling off to 0 past the tolerance
        band - a smooth target rather than a hard threshold."""
        z = self.data.qpos[2]
        return max(0.0, 1.0 - abs(z - TARGET_PALM_HEIGHT) / PALM_HEIGHT_TOLERANCE)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Physics changes go in after mj_resetData and before the state is set.
        self._randomize_physics()

        qpos = self._init_qpos.copy()
        # small random noise on joint angles only (not the freejoint) so
        # every episode doesn't start from an identical pose
        noise = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq - 7)
        qpos[7:] += noise
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0

        # Latency buffer starts filled with the neutral command (all zeros).
        self._action_buffer.clear()
        for _ in range(self._action_buffer.maxlen):
            self._action_buffer.append(np.zeros(self.nu, dtype=np.float64))
        self._shove_steps_left = 0
        self.data.xfrc_applied[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._elapsed_steps = 0
        self._last_action = np.zeros(self.nu)
        self._smoothed = np.zeros(self.nu)
        self._air_time[:] = 0.0
        self._since_contact[:] = 0.0
        _, self._tips_prev = self._floor_contacts()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Latency: the actuators execute the command from `latency` steps ago.
        self._smoothed += ACTION_SMOOTHING * (np.asarray(action, dtype=np.float64) - self._smoothed)
        self._action_buffer.append(self._smoothed.copy())
        delayed = self._action_buffer[-1 - self._latency]
        self.data.ctrl[:] = delayed

        self._update_shove()

        x_before = self.data.qpos[0]
        y_before = self.data.qpos[1]
        palm_touched = False
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            # Checked every substep, not just after the loop - a touch that
            # happens and clears within one control step would otherwise
            # go undetected.
            palm_now, tips_now = self._floor_contacts()
            palm_touched = palm_touched or palm_now
        x_after = self.data.qpos[0]
        y_after = self.data.qpos[1]

        forward_vel = (x_after - x_before) / (self.model.opt.timestep * FRAME_SKIP)
        forward_reward = FORWARD_REWARD_WEIGHT * forward_vel
        lateral_vel = (y_after - y_before) / (self.model.opt.timestep * FRAME_SKIP)
        lateral_cost = LATERAL_PENALTY_WEIGHT * abs(lateral_vel)
        control_cost = CONTROL_COST_WEIGHT * np.sum(np.square(action))
        action_rate_cost = ACTION_RATE_PENALTY_WEIGHT * np.sum(np.square(action - self._last_action))

        # Stepping: pay for each fingertip touchdown that follows a real swing
        # phase, but only while the hand is actually moving forward, so hopping
        # in place earns nothing. Then penalise any finger that sits out.
        first_touch = tips_now & ~self._tips_prev
        swing = np.clip(self._air_time[first_touch] - AIR_TIME_MIN, 0.0, AIR_TIME_CAP - AIR_TIME_MIN).sum()
        air_reward = AIR_TIME_WEIGHT * swing if forward_vel > 0.01 else 0.0
        dt = self._dt
        self._air_time = np.where(tips_now, 0.0, self._air_time + dt)
        self._since_contact = np.where(tips_now, 0.0, self._since_contact + dt)
        neglect = np.clip((self._since_contact - NEGLECT_AFTER_S) / NEGLECT_RAMP_S, 0.0, 1.0).mean()
        neglect_penalty = NEGLECT_PENALTY_WEIGHT * neglect
        self._tips_prev = tips_now
        self._last_action = np.asarray(action, dtype=np.float64)
        upright_reward = UPRIGHT_REWARD_WEIGHT * self._upright_reward()
        heading_reward = HEADING_REWARD_WEIGHT * self._heading_reward()
        height_reward = PALM_HEIGHT_WEIGHT * self._height_reward()
        palm_contact_penalty = PALM_CONTACT_PENALTY_WEIGHT * float(palm_touched)

        healthy = self._is_healthy()
        reward = (forward_reward + ALIVE_BONUS + upright_reward
                  + heading_reward + height_reward + air_reward
                  - control_cost - action_rate_cost - lateral_cost
                  - palm_contact_penalty - neglect_penalty)

        self._elapsed_steps += 1
        terminated = not healthy
        truncated = self._elapsed_steps >= MAX_EPISODE_STEPS

        info = {
            "forward_reward": forward_reward,
            "control_cost": control_cost,
            "upright_reward": upright_reward,
            "heading_reward": heading_reward,
            "height_reward": height_reward,
            "palm_contact_penalty": palm_contact_penalty,
            "air_reward": air_reward,
            "neglect_penalty": neglect_penalty,
            "action_rate_cost": action_rate_cost,
            "lateral_cost": lateral_cost,
            "tips_in_contact": int(tips_now.sum()),
            "x_position": x_after,
            "rand_scale": self._rand_scale,
            "latency_steps": self._latency,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=240, width=320)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
