<h1 align="center">16 DOF Direct-Driven Robot Hand</h1>

<h3 align="center">
  A robot hand <b>YOU</b> can build at home for less than <b>200 Euros</b>
</h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2e8fef0c-d946-41dc-bf10-3262e02fa41a" alt="Robot hand" width="700">
</p>

## Our Goal

**To democratize robotics hardware.**

Capable robot hands cost thousands of euros, which puts them out of reach for most hobbyists, students and independent researchers. This project is an open alternative: a 16 degree-of-freedom, direct-driven hand that you can build at home for **less than 200 €**, and that is good enough for hobby projects and for research.

Everything is built from parts you can order online. There are no custom-machined components and no special suppliers.

> ### Build it yourself
> **Every part is available to buy online, and every part is linked in the [Bill of Materials](BOM%20MY%20Robot%20Hand.csv).**
> Open the BOM, order the parts, and you have everything you need to build the hand.

## Highlights

- **16 degrees of freedom**: 4 fingers, with 4 independently driven joints each
- **Direct-driven**: one servo per joint, with no tendons or linkages
- **Metal-gear servos**: durable, and able to take the load of crawling
- **Under 200 €** in total, with every part linked in the BOM
- **Lightweight**: the complete hand weighs just **204 g**
- **Walks**: the hand crawls forward on its fingertips
- **Two simulation models**: an upright one for gestures and a floor-facing one for crawling
- **Control software included**: send gestures and the crawl to the real hand over USB serial

---

<h2 align="center">Gestures</h2>

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

### Videos

Watch the real hand perform each gesture, and crawl.

<table align="center">
  <tr>
    <td align="center" width="280"><b>Spread</b></td>
    <td align="center" width="280"><b>Spider-Man</b></td>
    <td align="center" width="280"><b>Peace</b></td>
  </tr>
  <tr>
    <td align="center" valign="middle" width="280" height="280"><video src="https://github.com/user-attachments/assets/2119add0-8ffd-4a32-9deb-a30aa2c67414" controls muted width="260" height="260"></video></td>
    <td align="center" valign="middle" width="280" height="280"><video src="https://github.com/user-attachments/assets/9fb2a27a-937c-461f-8101-f01ac08cd89b" controls muted width="260" height="260"></video></td>
    <td align="center" valign="middle" width="280" height="280"><video src="https://github.com/user-attachments/assets/fe2d028e-4e9b-4415-89fd-89c00d137b5e" controls muted width="260" height="260"></video></td>
  </tr>
  <tr>
    <td align="center" width="280"><b>Fist</b></td>
    <td align="center" width="280"><b>Counting</b></td>
    <td align="center" width="280"><b>Crawling</b></td>
  </tr>
  <tr>
    <td align="center" valign="middle" width="280" height="280"><video src="https://github.com/user-attachments/assets/5e618749-12ee-43e7-a32b-99904015908c" controls muted width="260" height="260"></video></td>
    <td align="center" valign="middle" width="280" height="280"><video src="https://github.com/user-attachments/assets/8af41ef5-231b-482b-b004-77c4e2cfe595" controls muted width="260" height="260"></video></td>
    <td align="center" valign="middle" width="280" height="280"><video src="https://github.com/user-attachments/assets/5c9b3b5d-ad53-4fb2-85c8-cd3dd8e16078" controls muted width="260" height="260"></video></td>
  </tr>
</table>

---

## It Can Walk

This hand isn't just for gestures. Flip it over so the palm faces the floor and the fingers become legs: the hand **crawls forward on its fingertips**.

> ⚠️ **Watch it closely.** The crawl runs open-loop, with no sensor feedback. Even metal-gear servos can overheat under sustained load, so keep an eye on them and keep your hand near the power switch.

### Try it

```bash
python send_gestures_to_hardware.py
# then type 4 and press Enter to start the crawl
# press Ctrl+C to stop it
```

---

## Weight

The complete hand weighs **204 g**.

Because the hand is direct-driven, every servo sits inside the fingers and palm. That makes weight important for mounting it on an arm and for how much load the crawl puts on the fingertip servos.

---

## Two Simulation Models

The hand comes with two MuJoCo models, depending on what you want to do:

| Model | Hand orientation | Use it for |
|---|---|---|
| `robothandassembly_fixed.xml` | **Upright** (fixed in place) | Gestures: fist, peace, spread and more |
| `robothandassembly_crawl.xml` | **Facing the floor** (free-moving, with a ground plane) | Crawling and walking experiments |

## What's in this repo

| Item | Description |
|---|---|
| [`BOM MY Robot Hand.csv`](BOM%20MY%20Robot%20Hand.csv) | Full parts list with purchase links |
| `robothandassembly_fixed.xml` | Simulation model, **upright hand** (gestures) |
| `robothandassembly_crawl.xml` | Simulation model, **hand facing the floor** (crawling) |
| `gesture_presets.py` | View and tune gestures in simulation |
| `send_gestures_to_hardware.py` | Send gestures and the crawl to the real hand |
| `pca9685_servo_bridge/` | Microcontroller firmware for driving the servos |
| `LICENSE` | License for the software |
| `LICENSE-HARDWARE` | License for the hardware designs and documentation |

---

## Future Improvements

This project is still growing. Planned updates will be added to this repo as they are built.

### Detailed assembly instructions

Step-by-step assembly instructions will be added, so that anyone can build the hand from the BOM without guessing. They are planned to cover:

- Assembling the fingers and palm
- Wiring the servos to the driver board
- Flashing the firmware and running the control software for the first time

### Live joint position feedback

Right now the hand runs **open-loop**: commands go out to the servos, but nothing reads back where the joints actually are. The plan is to read each servo's **internal potentiometer**, and to add an **I²C multiplexer** so that all 16 joints can be read over a single bus.

This would allow:

- **Live joint positions** streamed to the control software
- **Closed-loop control**, so the hand can check that it reached the pose it was told to reach
- **A safer crawl**, since the hand could react to what the joints are really doing instead of running blind

### Roadmap

- [ ] Write detailed step-by-step assembly instructions
- [ ] Read the internal potentiometer of each servo
- [ ] Add an I²C multiplexer so all 16 joints can be read over one bus
- [ ] Stream live joint positions to the control software
- [ ] Use the feedback for closed-loop control, including the crawl
- [ ] Update the BOM, firmware and instructions for the new parts

---

## License

This project is free for personal, hobby, educational and research use. **Commercial use is not permitted** without written permission.

- **Software** (Python scripts and firmware): [PolyForm Noncommercial 1.0.0](LICENSE)
- **Hardware** (CAD files, simulation models, BOM and documentation): [CC BY-NC 4.0](LICENSE-HARDWARE)

If you want to use this project commercially, contact the author through GitHub to ask for permission.
