import torch
import smplx
import trimesh
import numpy as np

# 1. Initialize model with flat hand mean (open palm as default)
model_path = '/home/luozhexi/data/hamer/_DATA/data/mano/MANO_RIGHT.pkl'
mano = smplx.MANO(model_path, is_rhand=True, flat_hand_mean=True, use_pca=False)

# 2. Forward pass with zero pose (flat hand)
hand_pose = torch.zeros(1, 45)
output = mano(hand_pose=hand_pose, return_verts=True, return_full_pose=True)
vertices = output.vertices.detach().cpu().numpy()[0]
joints = output.joints.detach().cpu().numpy()[0]
faces = mano.faces

# Print root joint coordinate (index 0 is wrist/root)
print(f"Root joint (wrist) coordinate: {joints[0]}")

# 3. Create hand mesh
hand_mesh = trimesh.Trimesh(vertices, faces, process=False)
hand_mesh.visual.face_colors = [200, 200, 250, 150]

# 4. Add coordinate axes at each joint
scene = trimesh.Scene([hand_mesh])
for i, joint_pos in enumerate(joints):
    axis = trimesh.creation.axis(origin_size=0.002, axis_radius=0.001, axis_length=0.02)
    axis.apply_translation(joint_pos)
    scene.add_geometry(axis)

# 5. Export to GLB
output_path = '/home/luozhexi/data/hamer/mano_hand.glb'
scene.export(output_path)
print(f"Exported to {output_path}")
