"""A pyrender-based helper for mesh or point cloud rendering.
Inspired from: https://github.com/davrempe/humor/blob/main/humor/viz/mesh_viewer.py
"""

import numpy as np
import pyrender
import trimesh


def create_raymond_lights():
    """
    Return raymond light nodes for the scene.
    """
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
        nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

    return nodes


# def make_look_at(position, target, up):
#     forward = np.subtract(target, position)
#     forward = np.divide(forward, np.linalg.norm(forward))
#     right = np.cross(forward, up)
#     # if forward and up vectors are parallel, right vector is zero;
#     #   fix by perturbing up vector a bit
#     if np.linalg.norm(right) < 0.001:
#         epsilon = np.array([0.001, 0, 0])
#         right = np.cross(forward, up + epsilon)
#     right = np.divide(right, np.linalg.norm(right))
#     up = np.cross(right, forward)
#     up = np.divide(up, np.linalg.norm(up))
#     return np.array(
#         [
#             [right[0], up[0], -forward[0], position[0]],
#             [right[1], up[1], -forward[1], position[1]],
#             [right[2], up[2], -forward[2], position[2]],
#             [0, 0, 0, 1],
#         ]
#     )


class Renderer(object):
    def __init__(self):
        pass

    def get_default_intrinsics(self, img_size, focal_length=5000.0):
        cx, cy = img_size[1] / 2, img_size[0] / 2
        intrinsics = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
        return intrinsics

    # def get_default_extrinsics(self):
    #     return make_look_at(position=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])

    def hack_transform_to_camera_frame(self, vertices, camera_extrinsics):
        # Somehow, camera extrinsics are not working as expected in pyrender.
        # As a hack, we transform the vertices to camera frame and use identity extrinsics.
        # This may have to do with the 180 degree rotation around x-axis.
        vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
        vertices = camera_extrinsics @ vertices.T
        vertices = vertices[:3, :].T
        vertices[..., 1:] *= -1  # I don't know why this 180-degree rotation around x is needed.
        return vertices, np.eye(4)

    def image_render(
        self,
        vertices,
        faces,
        image=None,
        camera_intrinsics=None,
        camera_extrinsics=None,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        alpha=1,
    ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
            camera_intrinsics (np.array): Array of shape (3, 3) containing the camera intrinsics.
            camera_extrinsics (np.array): Array of shape (3, 4) containing the camera extrinsics.
            image (np.array): Array of shape (H, W, 3) containing the input image.
            mesh_base_color (tuple): Tuple containing the base color of the mesh.
            scene_bg_color (tuple): Tuple containing the background color of the scene.
            alpha (float): Alpha value for blending the rendered image with the input image.
        """
        if image is None:
            image = np.tile(np.array(scene_bg_color).astype(np.uint8)[None, None, :], (360, 640))
        assert image.dtype == np.uint8, "Image should be a numpy array of type uint8"

        if camera_extrinsics is None:
            camera_extrinsics = np.eye(4)

        if camera_intrinsics is None:
            camera_intrinsics = self.get_default_intrinsics(img_size=image.shape[:2])

        if camera_extrinsics.shape[0] == 3:
            camera_extrinsics = np.concatenate([camera_extrinsics, np.array([[0, 0, 0, 1]])], axis=0)

        # renderer scene
        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1], viewport_height=image.shape[0])
        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # mesh
        vertices, camera_extrinsics = self.hack_transform_to_camera_frame(vertices, camera_extrinsics)
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        mesh = trimesh.Trimesh(vertices, faces.copy(), vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh, f"mesh")

        # camera
        camera = pyrender.IntrinsicsCamera(
            fx=camera_intrinsics[0, 0],
            fy=camera_intrinsics[1, 1],
            cx=camera_intrinsics[0, 2],
            cy=camera_intrinsics[1, 2],
            zfar=1e12,
        )
        camera_node = pyrender.Node(camera=camera, matrix=camera_extrinsics)
        scene.add_node(camera_node)

        # light
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        # render
        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        color = color.astype(np.float32) / 255.0
        image = image.astype(np.float32) / 255.0
        alpha = color[:, :, 3:] * alpha
        image = image[:, :, :3] * (1 - alpha) + color[:, :, :3] * alpha
        image = (image * 255).astype(np.uint8)
        return image
