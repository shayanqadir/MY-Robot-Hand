<h1 align="center">🦾 16 DOF Direct-Driven Robot Hand 🦾</h1>

<h3 align="center">
  A robot hand <b>YOU</b> can build at home for less than <b>200 Euros</b>
</h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2e8fef0c-d946-41dc-bf10-3262e02fa41a" alt="Robot hand" width="700">
</p>

<h2 align="center">✋ Gestures ✋</h2>

<table align="center">
  <tr>
    <td align="center" valign="bottom">
      <img src="https://github.com/user-attachments/assets/9dafd911-c15f-4b35-9e0f-a643205c650a" alt="Peace sign" height="180">
    </td>
    <td align="center" valign="bottom">
      <img src="https://github.com/user-attachments/assets/42e9811c-efed-490f-b8b4-cef22572473e" alt="Fist" height="180">
    </td>
    <td align="center" valign="bottom">
      <img src="https://github.com/user-attachments/assets/dbe04319-96c1-4ca9-a249-0d234965389e" alt="Spider-Man" height="180">
    </td>
    <td align="center" valign="bottom">
      <img src="https://github.com/user-attachments/assets/37bed6a0-0967-4bba-8622-5d14be7b15fd" alt="Spread" height="180">
    </td>
  </tr>
  <tr>
    <td align="center"><b>✌️ Peace</b></td>
    <td align="center"><b>✊ Fist</b></td>
    <td align="center"><b>🤟 Spider-Man</b></td>
    <td align="center"><b>🖐️ Spread</b></td>
  </tr>
</table>

---

## 🎬 Gesture Video

Watch the real hand move smoothly from one gesture to the next.

https://github.com/user-attachments/assets/4f824b5e-b6ed-4e0b-8b07-13fd044ababd

## 🎯 Our Goal

**To democratize robotics hardware.**

Capable robot hands cost thousands of euros, which puts them out of reach for most hobbyists, students and independent researchers. This project is an open alternative: a 16 degree-of-freedom, direct-driven hand that you can build at home for **less than 200 €**, and that is good enough for hobby projects and for research.

Everything is built from parts you can order online. There are no custom-machined components and no special suppliers.

> ### 🛒 Build it yourself
> **Every part is available to buy online, and every part is linked in the [Bill of Materials (BOM.md)](BOM.md).**
> Open the BOM, order the parts, and you have everything you need to build the hand.

## 🐾 It Can Walk!

This hand isn't just for gestures. Flip it over so the palm faces the floor and the fingers become legs: the hand **crawls forward on its fingertips**.

- **Hard-coded trot gait:** the walk is a hand-written oscillator, not a learned policy. The fingers move in diagonal pairs (index + pinky, and middle + thumb), with the two pairs half a cycle apart, like a trotting animal.
- **Three joints per leg:** the first joint swings the finger forward and back, the second lifts and pushes down, and the third adds a smaller secondary flex. Each runs a sine wave at a different phase.
- **Easy to tune:** stride speed, amplitude and phase offsets are plain constants at the top of the gait code, so you can experiment with your own gaits.

> ⚠️ **Watch it closely.** The gait runs open-loop, with no sensor feedback. Small plastic-gear servos can overheat under sustained load, so keep an eye on them and keep your hand near the power switch.

**Try it:**
```bash
python send_gestures_to_hardware.py
# then type 4 and press Enter to start the crawl gait
# press Enter again to stop it
```

## 🎥 Crawling Video

The hand crawling forward on its fingertips with the trot gait.

PASTE-CRAWL-VIDEO-LINK-HERE

## 🧭 Two Simulation Models

The hand comes with two MuJoCo models, depending on what you want to do:

| Model | Hand orientation | Use it for |
|---|---|---|
| `robothandassembly_fixed.xml` | **Upright** (fixed in place) | Gestures: fist, peace, spread and more |
| `robothandassembly_crawl.xml` | **Facing the floor** (free-moving, with a ground plane) | Crawling and walking experiments |

## ✨ Highlights

- **16 degrees of freedom**: 4 fingers, with 4 independently driven joints each
- **Direct-driven**: one servo per joint, with no tendons or linkages
- **Under 200 €** in total, with every part linked in the BOM
- **Two simulation models**: an upright one for gestures and a floor-facing one for crawling
- **Walks**: a hard-coded trot gait that crawls the hand forward on its fingertips
- **Control software included**: send gestures and the gait to the real hand over USB serial

## 📦 What's in this repo

| Item | Description |
|---|---|
| [`BOM.md`](My-Robot-Hand.csv) | Full parts list with purchase links |
| `robothandassembly_fixed.xml` | Simulation model, **upright hand** (gestures) |
| `robothandassembly_crawl.xml` | Simulation model, **hand facing the floor** (crawling) |
| `gesture_presets.py` | View and tune gestures in simulation |
| `send_gestures_to_hardware.py` | Send gestures and the trot gait to the real hand |
| `pca9685_servo_bridge/` | Microcontroller firmware for driving the servos |
| `hand_env.py`, `train_ppo.py`, `watch_policy.py`, `run_policy_on_hardware.py`, `sanity_check_model.py` | *Experimental:* reinforcement-learning work on the crawl model. Not used by the trot gait above. |
