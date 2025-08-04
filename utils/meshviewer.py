"""A pyrender-based helper for mesh or point cloud rendering.
Inspired from: https://github.com/davrempe/humor/blob/main/humor/viz/mesh_viewer.py
"""

import os, time, math

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
import sys
import cv2
import pyglet
from tqdm import tqdm
from pyrender.constants import RenderFlags


colors = {
    "pink": [0.7, 0.7, 0.9],
    "purple": [0.9, 0.7, 0.7],
    "cyan": [0.7, 0.75, 0.5],
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "yellow": [1.0, 1.0, 0],
    "brown": [0.5, 0.7, 0.7],
    "blue": [0.0, 0.0, 1.0],
    "offwhite": [0.8, 0.9, 0.9],
    "white": [1.0, 1.0, 1.0],
    "orange": [0.5, 0.65, 0.9],
    "grey": [0.7, 0.7, 0.7],
    "black": [0.0, 0.0, 0.0],
    "white": [1.0, 1.0, 1.0],
    "yellowg": [0.83, 1, 0],
}


def makeLookAt(position, target, up):
    forward = np.subtract(target, position)
    forward = np.divide(forward, np.linalg.norm(forward))

    right = np.cross(forward, up)

    # if forward and up vectors are parallel, right vector is zero;
    #   fix by perturbing up vector a bit
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array([0.001, 0, 0])
        right = np.cross(forward, up + epsilon)

    right = np.divide(right, np.linalg.norm(right))

    up = np.cross(right, forward)
    up = np.divide(up, np.linalg.norm(up))

    return np.array(
        [
            [right[0], up[0], -forward[0], position[0]],
            [right[1], up[1], -forward[1], position[1]],
            [right[2], up[2], -forward[2], position[2]],
            [0, 0, 0, 1],
        ]
    )


COMPRESS_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 9]


def pause_play_callback(pyrender_viewer, mesh_viewer):
    mesh_viewer.is_paused = not mesh_viewer.is_paused


def step_callback(pyrender_viewer, mesh_viewer, step_size):
    mesh_viewer.animation_frame_idx = (mesh_viewer.animation_frame_idx + step_size) % mesh_viewer.animation_len


class MeshViewer(object):
    def __init__(
        self,
        width=512,
        height=512,
        use_offscreen=True,
        camera_intrinsics=None,
        img_extn="png",
        cam_pos=[2.0, 2.0, 2.0],
        cam_lookat=[0.0, 0.0, 0.0],
        cam_up=[0.0, 1.0, 0.0],
        add_floor=False,
        add_axis=False,
    ):
        super().__init__()
        self.use_offscreen = use_offscreen
        # render settings for offscreen
        self.render_wireframe = False
        self.render_RGBA = False
        self.img_extn = img_extn

        # mesh sequences to animate
        self.animated_seqs = []  # the actual sequence of pyrender meshes
        self.animated_seqs_type = []
        self.animated_nodes = []  # the nodes corresponding to each sequence
        self.light_nodes = []
        # they must all be the same length (set based on first given sequence)
        self.animation_len = -1
        # current index in the animation sequence
        self.animation_frame_idx = 0

        # background image sequence
        self.img_seq = None
        self.cur_bg_img = None
        self.camera_seq = None

        self.single_frame = False

        self.scene = pyrender.Scene(bg_color=colors["white"], ambient_light=(0.3, 0.3, 0.3))

        self.default_cam_pose = makeLookAt(cam_pos, cam_lookat, cam_up)

        if camera_intrinsics is None:
            pc = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=float(width) / height)
            camera_pose = self.default_cam_pose.copy()
            self.camera_node = self.scene.add(pc, pose=camera_pose, name="pc-camera")

            light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            self.scene.add(light, pose=self.default_cam_pose)
        else:
            fx, fy, cx, cy = camera_intrinsics
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
            self.camera_node = self.scene.add(camera, pose=camera_pose, name="pc-camera")

            light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            self.scene.add(light, pose=camera_pose)

            self.set_background_color([1.0, 1.0, 1.0, 0.0])

        # key callbacks
        self.is_paused = False
        registered_keys = dict()
        registered_keys["p"] = (pause_play_callback, [self])
        registered_keys["."] = (step_callback, [self, 1])
        registered_keys[","] = (step_callback, [self, -1])

        if self.use_offscreen:
            # self.viewport_size = (width, height)
            self.viewer = pyrender.OffscreenRenderer(width, height)
            self.use_raymond_lighting(3.5)
        else:
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=(not camera_intrinsics),
                viewport_size=(width, height),
                cull_faces=False,
                run_in_thread=True,
                registered_keys=registered_keys,
                viewer_flags={
                    "rotate_axis": np.array([0.0, 1.0, 0.0]),
                    "show_world_axis": True,
                },
            )

        if add_floor:
            floor = trimesh.creation.box(10 * (1.001 - np.array(cam_up)))
            floor = trimesh.Trimesh(
                vertices=floor.vertices,
                faces=floor.faces,
                vertex_colors=floor.visual.vertex_colors,
            )
            self.add_static_meshes([floor])

        if add_axis:
            axs = trimesh.creation.axis()
            axs = trimesh.Trimesh(
                vertices=axs.vertices * 1,
                faces=axs.faces,
                vertex_colors=axs.visual.vertex_colors,
            )
            self.add_static_meshes([axs])

    def set_background_color(self, color=colors["white"]):
        self.scene.bg_color = color

    def update_camera_pose(self, camera_pose):
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def update_camera(self, cam):
        self.scene.remove_node(self.camera_node)
        yfov = cam["yfov"]
        width = cam["width"]
        height = cam["height"]
        pc = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=float(width) / height)
        camera_pose = cam["T"]
        self.camera_node = self.scene.add(pc, pose=camera_pose, name="pc-camera")

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_render_settings(self, wireframe=None, RGBA=None, single_frame=None):
        if wireframe is not None and wireframe == True:
            self.render_wireframe = True
        if RGBA is not None and RGBA == True:
            self.render_RGBA = True
        if single_frame is not None:
            self.single_frame = single_frame

    def acquire_render_lock(self):
        if not self.use_offscreen:
            self.viewer.render_lock.acquire()

    def release_render_lock(self):
        if not self.use_offscreen:
            self.viewer.render_lock.release()

    def _add_raymond_light(self):
        from pyrender.light import DirectionalLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                Node(
                    light=DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )
        return nodes

    def use_raymond_lighting(self, intensity=1.0):
        if not self.use_offscreen:
            sys.stderr.write("Interactive viewer already uses raymond lighting!\n")
            return
        for n in self._add_raymond_light():
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)  # , parent_node=pc)

            self.light_nodes.append(n)

    # ----------------------------------------------
    # Static meshes
    # ----------------------------------------------
    def set_meshes(self, meshes, group_name="static", remove_old=False):
        """Add a list of static meshes to the scene."""
        if remove_old:
            for node in self.scene.get_nodes():
                if node.name is not None and "%s-mesh" % group_name in node.name:
                    self.scene.remove_node(node)

        for mid, mesh in enumerate(meshes):
            if isinstance(mesh, trimesh.Trimesh):
                mesh = pyrender.Mesh.from_trimesh(mesh.copy())
            self.acquire_render_lock()
            self.scene.add(mesh, "%s-mesh-%2d" % (group_name, mid))
            self.release_render_lock()

    def set_static_meshes(self, meshes):
        self.set_meshes(meshes, group_name="static", remove_old=True)

    def add_static_meshes(self, meshes):
        self.set_meshes(meshes, group_name="staticadd", remove_old=False)

    # ----------------------------------------------
    # Animation assets
    # ----------------------------------------------
    def check_animation_len(self, cur_seq_len):
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print("Unexpected imgage sequence length, all sequences must be the same length!")
                assert False
                return False
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
                return True
            else:
                print("Warning: imge sequence is length 0!")
                assert False
                return False
        return True

    def set_img_seq(self, img_seq):
        """np array of BG images to be rendered in background."""
        if not self.use_offscreen:
            print("Cannot render background image if not rendering offscreen")
            return
        if not self.check_animation_len(len(img_seq)):
            return
        self.img_seq = img_seq
        # must have alpha to render background
        self.set_render_settings(RGBA=True)

    def add_pyrender_mesh_seq(self, pyrender_mesh_seq, seq_type="default"):
        if not self.check_animation_len(len(pyrender_mesh_seq)):
            return
        # add to the list of sequences to render
        seq_id = len(self.animated_seqs)
        self.animated_seqs.append(pyrender_mesh_seq)
        self.animated_seqs_type.append(seq_type)
        # create the corresponding node in the scene
        self.acquire_render_lock()
        anim_node = self.scene.add(pyrender_mesh_seq[0], "anim-mesh-%2d" % (seq_id))
        self.animated_nodes.append(anim_node)
        self.release_render_lock()

    def add_mesh_seq(self, mesh_seq):
        """Add a sequence of trimesh.trimesh objects for every frame to be rendered."""
        if not self.check_animation_len(len(mesh_seq)):
            return
        # print('Adding mesh sequence with %d frames...' % (len(mesh_seq)))

        pyrender_mesh_seq = []
        # for mid, mesh in tqdm(enumerate(mesh_seq), desc='Caching meshes'):
        for mid, mesh in enumerate(mesh_seq):
            if isinstance(mesh, trimesh.Trimesh):
                mesh = pyrender.Mesh.from_trimesh(mesh.copy())
                pyrender_mesh_seq.append(mesh)
            else:
                print("Meshes must be from trimesh!")
                return
        self.add_pyrender_mesh_seq(pyrender_mesh_seq, seq_type="mesh")

    def add_point_seq(self, point_seq, color=[1.0, 0.0, 0.0], radius=0.02):
        """Add a sequence of pointclouds to be rendered as spheres."""

        if not self.check_animation_len(len(point_seq)):
            return
        sm = trimesh.creation.uv_sphere(radius=radius, count=[4, 4])
        sm.visual.vertex_colors = color
        pyrender_point_seq = []
        # for pid, points in tqdm(enumerate(point_seq), desc='Caching point meshes'):
        for pid, points in enumerate(point_seq):
            tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
            tfs[:, :3, 3] = points
            pyrender_point_seq.append(pyrender.Mesh.from_trimesh(sm.copy(), poses=tfs))
        self.add_pyrender_mesh_seq(pyrender_point_seq, seq_type="point")

    def add_camera_seq(self, camera_seq):
        if not self.check_animation_len(len(camera_seq)):
            return
        self.camera_seq = camera_seq

    def add_line_seq(self, line_seq, color=[0.0, 0.0, 0.0]):
        """Add a sequence of lines to be rendered."""
        if not self.check_animation_len(len(line_seq)):
            return
        pyrender_line_seq = []
        # for lid, lines in tqdm(enumerate(line_seq), desc='Caching lines'):
        for lid, lines in enumerate(line_seq):
            pyrender_line_seq.append(
                pyrender.Mesh([pyrender.Primitive(lines, mode=pyrender.constants.GLTF.LINES, color_0=color)])
            )
        self.add_pyrender_mesh_seq(pyrender_line_seq, seq_type="line")

    # ----------------------------------------------
    # Animation and Rendering
    # ----------------------------------------------

    def render(self):
        """Render into an array."""
        # flags = RenderFlags.SHADOWS_DIRECTIONAL
        flags = RenderFlags.NONE
        flags |= RenderFlags.SKIP_CULL_FACES
        if self.render_RGBA:
            flags |= RenderFlags.RGBA
        if self.render_wireframe:
            flags |= RenderFlags.ALL_WIREFRAME
        color_img, depth_img = self.viewer.render(self.scene, flags=flags)
        output_img = color_img
        if self.cur_bg_img is not None:
            color_img = color_img.astype(np.float32) / 255.0
            person_mask = None
            if self.cur_mask is not None:
                person_mask = self.cur_mask[:, :, np.newaxis]
                color_img = color_img * (1.0 - person_mask)
            valid_mask = (color_img[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = self.cur_bg_img
            if color_img.shape[2] == 4:
                output_img = color_img[:, :, :-1] * color_img[:, :, 3:] + (1.0 - color_img[:, :, 3:]) * input_img
            else:
                output_img = color_img[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img
            output_img = (output_img * 255.0).astype(np.uint8)

        return output_img

    def save_image(self, color_img, fname):
        if not self.use_offscreen:
            sys.stderr.write("Currently saving snapshots only works with off-screen renderer!\n")
            return
        if color_img.shape[-1] == 4:
            img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGRA)
        else:
            img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fname, img_bgr, COMPRESS_PARAMS)

    def update_frame(self):
        """Update frame to show the current self.animation_frame_idx"""
        for seq_idx in range(len(self.animated_seqs)):
            cur_mesh = self.animated_seqs[seq_idx][self.animation_frame_idx]
            # render the current frame of eqch sequence
            self.acquire_render_lock()

            # replace the old mesh
            anim_node = list(self.scene.get_nodes(name="anim-mesh-%2d" % (seq_idx)))[0]
            anim_node.mesh = cur_mesh
            self.release_render_lock()

        # update camera pose
        if self.camera_seq is not None:
            self.update_camera(self.camera_seq[self.animation_frame_idx])

        # update background img
        if self.img_seq is not None:
            self.acquire_render_lock()
            self.cur_bg_img = self.img_seq[self.animation_frame_idx]
            self.release_render_lock()

    def animate(self, fps=30, save_path=None):
        """
        Starts animating any given mesh sequences. This should be called last after adding
        all desired components to the scene as it is a blocking operation and will run
        until the user exits (or the full video is rendered if offline).
        """
        if not self.use_offscreen:
            print("=================================")
            print("VIEWER CONTROLS")
            print("p - pause/play")
            print('"," and "." - step back/forward one frame')
            print("w - wireframe")
            print("h - render shadows")
            print("q - quit")
            print("=================================")

        frame_dur = 1.0 / float(fps)
        seq_imgs = []

        # set up init frame
        self.update_frame()
        animation_render_time = time.time()

        while self.use_offscreen or self.viewer.is_active:
            if not self.use_offscreen:
                sleep_len = frame_dur - (time.time() - animation_render_time)
                if sleep_len > 0:
                    time.sleep(sleep_len)
            else:
                # render frame
                img = self.render()
                seq_imgs.append(img)

                if save_path:
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                        print(f"Rendering frames to {save_path}!")

                    cur_file_path = os.path.join(
                        save_path,
                        "frame_%08d.%s" % (self.animation_frame_idx, self.img_extn),
                    )
                    self.save_image(img, cur_file_path)

                if self.animation_frame_idx + 1 >= self.animation_len:
                    self.viewer.delete()
                    break

            animation_render_time = time.time()
            if self.is_paused:
                self.update_frame()  # just in case there's a single frame update
                continue

            self.animation_frame_idx = (self.animation_frame_idx + 1) % self.animation_len
            self.update_frame()

            if self.single_frame:
                break

        self.animation_frame_idx = 0
        return np.stack(seq_imgs)
