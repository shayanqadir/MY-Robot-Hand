"""
Compute the REAL fingertip contact points from the distal-phalanx mesh
vertices, and print ready-to-paste <geom> lines for the MJCF.

    python find_fingertips.py

Why this exists: the tip spheres in robothandassembly_crawl.xml were placed
at pos="0 0.02 0", which was a guess written in the 2GMOTOR_3-style body
frame. That lands on the last JOINT, not on the end of the last phalanx, so
the distal link hangs past the sphere with no collision on it at all and
punches straight through the floor when the finger curls.

This script reads the actual STL geometry you already have in meshes/ and
finds, for each finger, the vertex of the distal phalanx that sits farthest
from that finger's last joint axis. That point is the fingertip.

NOTE ON BODY CHOICE: NEWNEWTHUMB_* and NEWNEWTHUMBOTHERHALF_* have no
<joint>, so MuJoCo welds them to their parent 2GMOTOR_* body -- they are all
one rigid body at simulation time. That means it does NOT matter which of
them the sphere lives in; only the position matters. So this script keeps
the sphere in its current parent body and just corrects pos, which is the
smallest possible change to your model.
"""

import numpy as np
import mujoco

MODEL_PATH = "robothandassembly_crawl.xml"

# finger -> (body holding the tip sphere, last joint, distal phalanx meshes)
FINGERS = {
    "thumb":  ("ThumbFinger_1__2GMOTOR_15",  "ThumbJoint3",
               ["ThumbFinger_1__NEWNEWTHUMB_4",  "ThumbFinger_1__NEWNEWTHUMBOTHERHALF_4"]),
    "index":  ("IndexFinger_1__2GMOTOR_3",   "IndexJoint3",
               ["IndexFinger_1__NEWNEWTHUMB_1",  "IndexFinger_1__NEWNEWTHUMBOTHERHALF_1"]),
    "middle": ("MiddleFinger_1__2GMOTOR_7",  "MiddleJoint3",
               ["MiddleFinger_1__NEWNEWTHUMB_2", "MiddleFinger_1__NEWNEWTHUMBOTHERHALF_2"]),
    "pinky":  ("PinkyFinger_1__2GMOTOR_11",  "PinkyJoint3",
               ["PinkyFinger_1__NEWNEWTHUMB_3", "PinkyFinger_1__NEWNEWTHUMBOTHERHALF_3"]),
}

SPHERE_RADIUS = 0.004


def mesh_vertices_world(model, data, geom_id):
    """All vertices of a mesh geom, in world coordinates."""
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return None
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[start:start + count]          # geom-local frame
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    return xpos + verts @ xmat.T


def geoms_of_body(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return bid, []
    return bid, [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)   # qpos=0 rest pose

    print(f"{'finger':8s} {'tip pos in parent body frame':34s} {'world z':>9s}  "
          f"{'dist from joint':>15s}")
    print("-" * 76)

    new_lines = {}
    capsule_lines = {}
    for finger, (parent_body, joint_name, distal_bodies) in FINGERS.items():
        pbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_body)
        if pbid < 0:
            print(f"{finger}: body '{parent_body}' not found - check the name")
            continue

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_world = data.xanchor[jid] if jid >= 0 else data.xpos[pbid]

        # gather every vertex of the distal phalanx halves
        all_verts = []
        for db in distal_bodies:
            bid, gids = geoms_of_body(model, db)
            if bid < 0:
                print(f"  (warning) distal body '{db}' not found")
                continue
            for g in gids:
                v = mesh_vertices_world(model, data, g)
                if v is not None:
                    all_verts.append(v)

        if not all_verts:
            print(f"{finger}: no distal mesh vertices found")
            continue

        verts = np.vstack(all_verts)

        # the tip = vertex farthest from the last joint anchor
        d = np.linalg.norm(verts - joint_world, axis=1)
        tip_world = verts[np.argmax(d)]
        tip_dist = d.max()

        # express it in the parent body's frame, where the <geom pos> lives
        pbody_pos = data.xpos[pbid]
        pbody_mat = data.xmat[pbid].reshape(3, 3)
        tip_local = pbody_mat.T @ (tip_world - pbody_pos)

        # pull the sphere back inside the mesh by its own radius so the
        # surface of the sphere sits at the surface of the fingertip
        inward = (joint_world - tip_world)
        inward /= np.linalg.norm(inward)
        tip_local_adj = pbody_mat.T @ (tip_world + inward * SPHERE_RADIUS - pbody_pos)

        pos_str = " ".join(f"{c:.5f}" for c in tip_local_adj)
        print(f"{finger:8s} {pos_str:34s} {tip_world[2]:9.4f}  {tip_dist:15.4f}")

        new_lines[finger] = (
            f'<geom name="{finger}_tip_collision" type="sphere" '
            f'size="{SPHERE_RADIUS}" pos="{pos_str}" '
            f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001" '
            f'rgba="1 0 0 0.5"/>'
        )

        # A lone sphere at the very tip still leaves the ~2-3cm phalanx SHAFT
        # with no collision, so a hard curl can still let the middle of the
        # last link sink through the floor. A capsule from the joint to the
        # tip covers the whole link and is barely more expensive to simulate.
        joint_local = pbody_mat.T @ (joint_world - pbody_pos)
        fromto = " ".join(f"{c:.5f}" for c in np.concatenate([joint_local, tip_local_adj]))
        capsule_lines[finger] = (
            f'<geom name="{finger}_distal_collision" type="capsule" '
            f'size="{SPHERE_RADIUS}" fromto="{fromto}" '
            f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001" '
            f'rgba="1 0 0 0.5"/>'
        )

    print("\nOPTION A - corrected tip spheres. Replace each existing "
          "*_tip_collision line with:\n")
    for finger, line in new_lines.items():
        print(f"  [{finger}]  {line}")

    print("\nOPTION B (recommended) - capsules covering the whole distal "
          "phalanx. Use these INSTEAD of Option A:\n")
    for finger, line in capsule_lines.items():
        print(f"  [{finger}]  {line}")

    print(
        "\nSanity check before you trust these: the old pos was '0 0.02 0'. "
        "If the numbers above are wildly different in magnitude, that's "
        "expected - that's the bug. But open the viewer afterwards, press C "
        "and T, and confirm each red sphere visually sits on the fingertip "
        "surface."
    )


if __name__ == "__main__":
    main()
