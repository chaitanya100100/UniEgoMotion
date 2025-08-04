import numpy as np
import trimesh
import os
import cv2
import tempfile
from tqdm.auto import tqdm
import torch

from .torch_utils import to_numpy
from .meshviewer import MeshViewer


def visualize_sequence(
    *,
    aria_traj=None,
    verts=None,
    global_jpos=None,
    global_rotmat=None,
    pred_aria_traj=None,
    pred_verts=None,
    pred_global_jpos=None,
    floor_height=0.0,
    faces=None,
    mv_params=None,
):
    # Setup MeshViewer
    center = np.array([0, 0, 0])
    default_mv_params = dict(
        width=1440 // 2,
        height=1080 // 2,
        use_offscreen=True,
        # cam_pos=center + np.array([-3, -3, 1]),
        cam_pos=center + np.array([-2, -7, 1]),
        cam_lookat=center + np.array([0, 0, 1]),
        cam_up=[0.0, 0.0, 1.0],
        add_axis=True,
        add_floor=True,
    )
    if mv_params is not None:
        default_mv_params.update(mv_params)
    mv = MeshViewer(**default_mv_params)

    pred_color = np.array([64, 128, 192]).astype(np.uint8)
    default_color = np.array([128, 128, 128]).astype(np.uint8)

    # Add stuff to MeshViewer
    offset = np.array([0, 0, floor_height])[None]
    if aria_traj is not None:
        aria_traj_seq = []
        aria_traj = to_numpy(aria_traj)
        for at in aria_traj:
            ms = trimesh.creation.axis(axis_length=0.2, origin_color=default_color).apply_transform(at)
            ms = trimesh.Trimesh(vertices=ms.vertices - offset, faces=ms.faces, vertex_colors=ms.visual.vertex_colors)
            aria_traj_seq.append(ms)
        mv.add_mesh_seq(aria_traj_seq)

    if pred_aria_traj is not None:
        pred_aria_traj_seq = []
        pred_aria_traj = to_numpy(pred_aria_traj)
        for pat in pred_aria_traj:
            ms = trimesh.creation.axis(axis_length=0.2, origin_color=pred_color).apply_transform(pat)
            ms = trimesh.Trimesh(vertices=ms.vertices - offset, faces=ms.faces, vertex_colors=ms.visual.vertex_colors)
            pred_aria_traj_seq.append(ms)
        mv.add_mesh_seq(pred_aria_traj_seq)

    if verts is not None:
        assert faces is not None
        mesh_seq = []
        verts = to_numpy(verts)
        faces = to_numpy(faces)
        for v in verts:
            ms = trimesh.Trimesh(vertices=v - offset, faces=faces)
            ms.visual.vertex_colors[:, -1] = 128
            ms.visual.vertex_colors[:, :-1] = default_color[None, :]
            mesh_seq.append(ms)
        mv.add_mesh_seq(mesh_seq)

    if pred_verts is not None:
        assert faces is not None
        pred_mesh_seq = []
        pred_verts = to_numpy(pred_verts)
        faces = to_numpy(faces)
        for pv in pred_verts:
            ms = trimesh.Trimesh(vertices=pv - offset, faces=faces)
            ms.visual.vertex_colors[:, -1] = 128
            ms.visual.vertex_colors[:, :-1] = pred_color[None, :]
            pred_mesh_seq.append(ms)
        mv.add_mesh_seq(pred_mesh_seq)

    if global_jpos is not None:
        global_jpos_seq = []
        global_jpos = to_numpy(global_jpos)
        for j in global_jpos:
            global_jpos_seq.append(j - offset)
        mv.add_point_seq(global_jpos_seq)

    if pred_global_jpos is not None:
        pred_global_jpos_seq = []
        pred_global_jpos = to_numpy(pred_global_jpos)
        for pj in pred_global_jpos:
            pred_global_jpos_seq.append(pj - offset)
        mv.add_point_seq(pred_global_jpos_seq, color=[0, 0, 1])

    if global_rotmat is not None:
        assert global_jpos is not None
        global_rotmat = to_numpy(global_rotmat)
        global_jpos = to_numpy(global_jpos)
        for i in range(min(22, global_rotmat.shape[1])):
            jrot_seq = []
            for j in range(len(global_jpos)):
                tsfm = np.eye(4)
                tsfm[:3, :3] = global_rotmat[j, i]
                tsfm[:3, 3] = global_jpos[j, i]
                ms = trimesh.creation.axis(axis_length=0.2).apply_transform(tsfm)
                ms = trimesh.Trimesh(
                    vertices=ms.vertices - offset, faces=ms.faces, vertex_colors=ms.visual.vertex_colors
                )
                jrot_seq.append(ms)

            mv.add_mesh_seq(jrot_seq)

    # Render
    imgs = mv.animate()
    return imgs
    # print(imgs.shape)
    # save_video(imgs[..., ::-1], f"traj", "/vision/u/chpatel/test", fps=10)


def visualize_sequence_blender(
    *,
    aria_traj=None,
    verts=None,
    global_jpos=None,
    global_rotmat=None,
    pred_aria_traj=None,
    pred_verts=None,
    pred_global_jpos=None,
    floor_height=0.0,
    faces=None,
    camera_params=None,
    render_params=None,
    max_num_frames=None,
    scene_blend_file="./assets/floor.blend",
):
    from blendify import scene
    from blendify.colors import UniformColors, VertexColors
    from blendify.materials import PrincipledBSDFMaterial
    from blendify.utils.image import blend_with_background

    # Load the scene
    scene.clear()
    scene.attach_blend(scene_blend_file)
    # scene.lights.add_sun(strength=5, translation=(0, 10, 10))

    default_camera_params = dict(
        resolution=(1440 // 2, 1080 // 2),
        fov_x=np.deg2rad(45),
        rotation=(0.0, 0.0, 0.5),
        rotation_mode="look_at",
        # translation=[3.5, -3.5, 2],
        translation=[-2, -7, 1],
        tracking_camera=False,
    )
    if camera_params is not None:
        default_camera_params.update(camera_params)

    tracking_camera = default_camera_params["tracking_camera"]
    camera = scene.set_perspective_camera(
        default_camera_params["resolution"],
        fov_x=default_camera_params["fov_x"],
        rotation=default_camera_params["rotation"],
        rotation_mode=default_camera_params["rotation_mode"],
        translation=default_camera_params["translation"],
    )

    default_color = (0.5, 0.5, 0.5)
    pred_color = (0.25, 0.5, 0.75)

    material = PrincipledBSDFMaterial(metallic=0.1)
    transparent_material = PrincipledBSDFMaterial(metallic=0.1)
    offset = np.array([0, 0, floor_height])[None]

    def resolve_nframes(nframes, cur_nframes):
        if nframes is None:
            nframes = cur_nframes
        else:
            assert nframes == cur_nframes
        return nframes

    nframes = None
    xy_mask = np.array([1, 1, 0]).astype(np.float32)
    centers = []

    def get_axis_mesh(transform, color):
        axis_mesh = trimesh.creation.axis(axis_length=0.2, origin_color=color).apply_transform(transform)
        return axis_mesh

    aria_traj_mesh = None
    if aria_traj is not None:
        axis_mesh = get_axis_mesh(to_numpy(aria_traj[0]), default_color)
        aria_traj_mesh = scene.renderables.add_mesh(
            axis_mesh.vertices - offset,
            axis_mesh.faces,
            material=material,
            colors=VertexColors(axis_mesh.visual.vertex_colors.astype(np.float32) / 255),
        )
        nframes = resolve_nframes(nframes, len(aria_traj))

        if tracking_camera:
            centers.append((to_numpy(aria_traj[:, :3, 3]) - offset) * xy_mask)

    pred_aria_traj_mesh = None
    if pred_aria_traj is not None:
        axis_mesh = get_axis_mesh(to_numpy(pred_aria_traj[0]), pred_color)
        pred_aria_traj_mesh = scene.renderables.add_mesh(
            axis_mesh.vertices - offset,
            axis_mesh.faces,
            material=material,
            colors=VertexColors(axis_mesh.visual.vertex_colors.astype(np.float32) / 255),
        )
        nframes = resolve_nframes(nframes, len(pred_aria_traj))

        if tracking_camera:
            centers.append((to_numpy(pred_aria_traj[:, :3, 3]) - offset) * xy_mask)

    verts_mesh = None
    if verts is not None:
        assert faces is not None
        verts_mesh = scene.renderables.add_mesh(
            to_numpy(verts[0]) - offset,
            to_numpy(faces),
            material=transparent_material,
            colors=UniformColors(default_color),
        )
        nframes = resolve_nframes(nframes, len(verts))

        if tracking_camera:
            centers.append(to_numpy(verts.mean(1) - offset) * xy_mask)

    pred_verts_mesh = None
    if pred_verts is not None:
        assert faces is not None
        pred_verts_mesh = scene.renderables.add_mesh(
            to_numpy(pred_verts[0]) - offset,
            to_numpy(faces),
            material=transparent_material,
            colors=UniformColors(pred_color),
        )
        nframes = resolve_nframes(nframes, len(pred_verts))

        if tracking_camera:
            centers.append(to_numpy(pred_verts.mean(1) - offset) * xy_mask)

    assert global_jpos is None
    assert pred_global_jpos is None
    assert global_rotmat is None

    default_render_params = dict(use_gpu=True, samples=128, use_denoiser=False)
    if render_params is not None:
        default_render_params.update(render_params)

    if tracking_camera:
        centers = np.stack(centers).mean(0)
        camera_offset = np.array(default_camera_params["translation"])

    # Rendering loop
    imgs = []
    for frame in tqdm(range(nframes)):
        if max_num_frames is not None and frame >= max_num_frames:
            break

        if tracking_camera:
            camera.set_position(translation=camera_offset + centers[frame])

        if aria_traj is not None:
            axis_mesh = get_axis_mesh(to_numpy(aria_traj[frame]), default_color)
            aria_traj_mesh.update_vertices(axis_mesh.vertices - offset)

        if pred_aria_traj is not None:
            axis_mesh = get_axis_mesh(to_numpy(pred_aria_traj[frame]), pred_color)
            pred_aria_traj_mesh.update_vertices(axis_mesh.vertices - offset)

        if verts is not None:
            verts_mesh.update_vertices(to_numpy(verts[frame]) - offset)
        if pred_verts is not None:
            pred_verts_mesh.update_vertices(to_numpy(pred_verts[frame]) - offset)

        img = scene.render(**default_render_params)
        img_white_bkg = blend_with_background(img, (1.0, 1.0, 1.0))
        imgs.append(img_white_bkg)

    imgs = np.stack(imgs, axis=0)
    scene.clear()
    return imgs


def save_video(images, video_name, out_dir, fps=30):
    """
    Renders a video from a sequence of images.
    """
    # save as images in tmp directory and run ffmpeg

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(images):
            cv2.imwrite(f"{tmpdir}/{i:06d}.jpg", img)
        cmd = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{tmpdir}/*.jpg' -c:v libx264 -pix_fmt yuv420p '{out_dir}/{video_name}.mp4'"
        os.system(cmd)
    print(f"Created {video_name}.mp4")

    # H, W, C = images.shape[1:]
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
    # video = cv2.VideoWriter(f"{out_dir}/{video_name}.mp4", fourcc, fps, (W, H))

    # for image in images:
    #     image = np.uint8(image)
    #     video.write(image)

    # # Release the video writer
    # video.release()
    # print(f"Created {video_name}.mp4")


def put_text_on_img(
    img_inp,
    text,
    font_scale=3,
    pos=(0, 0),
    font_thickness=2,
    text_color=(255, 255, 255),
    text_color_bg=(0, 0, 0),
    font=cv2.FONT_HERSHEY_PLAIN,
):
    # img_np can be either a numpy array or a torch.Tensor. It can be a single image or a batch of images.
    # Handle torch.Tensor input
    if isinstance(img_inp, torch.Tensor):
        img = img_inp.detach().cpu().numpy()
    else:
        img = img_inp

    # Handle batched inputs
    if len(img_inp.shape) == 4:
        img = np.stack(
            [put_text_on_img(x, text, font_scale, pos, font_thickness, text_color, text_color_bg, font) for x in img]
        )
        if isinstance(img_inp, torch.Tensor):
            img = torch.from_numpy(img).to(img_inp.dtype).to(img_inp.device)
        return img

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if text_color_bg is not None:
        img = cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    img = cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    if isinstance(img_inp, torch.Tensor):
        img = torch.from_numpy(img).to(img_inp.dtype).to(img_inp.device)
    return img


def pad_filler(verts, nf):
    # verts : T x V x 3
    pad_nf = nf - verts.shape[0]
    if pad_nf > 0:
        filler = torch.ones_like(verts[-1:]) * (-100)  # send it far from rendering frustum
        verts = torch.cat([verts, filler.expand(pad_nf, -1, -1)], dim=0)
    assert verts.shape[0] == nf
    return verts


def pad_filler_traj(traj, nf):
    # traj : T x 3
    pad_nf = nf - traj.shape[0]
    if pad_nf > 0:
        filler = traj[-1:] * (-100)  # send it far from rendering frustum
        traj = torch.cat([traj, filler.expand(pad_nf, -1, -1)], dim=0)
    assert traj.shape[0] == nf
    return traj
