import os
import pickle as pkl
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.renderer.mesh.utils import _clip_barycentric_coordinates as clip_barycentric_coordinates
import cv2
from matplotlib import cm
from pytorch3d.renderer import (
    PerspectiveCameras,
)
import pandas as pd
from scipy import signal
import skimage
from scipy.spatial.transform import Rotation as R

workspace_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
import sys
sys.path.append(workspace_path)

import lib.constants as constants

def filter_signal(recording, savgol_window=11, savgol_polyorder=3, axis=0, deriv=0):
    return signal.savgol_filter(recording, window_length=savgol_window, polyorder=savgol_polyorder, axis=axis, deriv=axis)

## Frame normalized and wraped around
def get_frame_for_display(frame, min_value=None, max_value=None):
    if min_value is None or max_value is None:
        min_value = np.min(frame)
        max_value = np.max(frame)
    frame_normalized = (frame - min_value) / (max_value - min_value)
    # Apply the hot colormap
    frame_colormap = cm.viridis(frame_normalized)
    # Convert the colormap to 8-bit unsigned integer format
    frame_uint8 = (frame_colormap * 255).astype(np.uint8)
    # # Convert the colormap to BGR for OpenCV display
    # frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

    return frame_uint8


## Load and save mesh
def save_meshes_pkl(v, f, workspace_path, filename, exp_name, **kwargs):
    # create a folder named as filename in results folder
    # save the pkl file in that folder
    os.makedirs(os.path.join(workspace_path, 'results', filename), exist_ok=True)
    dict_to_save = dict(
        verts_seq = v,
        faces = f,
    )
    dict_to_save.update(kwargs)
    with open(os.path.join(workspace_path, 'results', filename, exp_name + '.pkl'), 'wb') as f:
        pkl.dump(dict_to_save, f)

def load_meshes_pkl(workspace_path, filename, exp_name):
    with open(os.path.join(workspace_path, 'results', filename, exp_name + '.pkl'), 'rb') as f:
        dict_to_load = pkl.load(f)
    return dict_to_load


## plot n images
def plt_plot_first_nimages(images, n=1, num_cols=4, figsize=(10, 5), title=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if title is not None:
        plt.suptitle(title)
    print(min(images.shape[0], n))
    for i in range(min(images.shape[0], n)):
        plt.subplot(n//num_cols+1, num_cols, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

## Load Mesh From Object Name
def get_vf_for_obj(obj_name, device, norm=True, full_obj_path=None):
    global workspace_path
    if full_obj_path is None:
        verts, faces, aux = load_obj(os.path.join(
            workspace_path, f'objs/{obj_name}.obj'), device=device)
    else:
        verts, faces, aux = load_obj(full_obj_path, device=device)

    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    if norm:
        center = (verts.max(0).values + verts.min(0).values) / 2
        verts = verts - center
        scale = max(verts.abs().max(0)[0][:2])
        verts = verts / scale

    print(verts.min(0)[0], verts.max(0)[0])
    return verts, faces_idx



## Counts to Temp
def counts2temp(data_counts, camera):
    Emiss = constants.EPSILON
    K1 = 1 / (constants.Tau * Emiss * constants.TransmissionExtOptics)
        
    R = constants.R_const[camera]
    B = constants.B_const[camera]
    F = constants.F_const[camera]
    O = constants.O_const[camera]
    
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/constants.TRefl)-F))
    # # Pseudo radiance of the atmosphere
    r2 = ((1 - constants.Tau)/(Emiss * constants.Tau)) * (R/(np.exp(B/constants.TAtm)-F)) 
    # # Pseudo radiance of the external optics
    r3 = ((1-constants.TransmissionExtOptics) / (Emiss * constants.Tau * constants.TransmissionExtOptics)) * (R/(np.exp(B/constants.TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = data_counts.astype(float)
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2 - O) + F))
    # print(np.amin(data_counts), np.amax(data_counts), np.amin(data_temp), np.amax(data_temp), data_obj_signal.shape)
    
    return data_temp
## Temp to Counts
def temp2counts(temp, camera):
    Emiss = constants.EPSILON
    K1 = 1 / (constants.Tau * Emiss * constants.TransmissionExtOptics)
        
    R = constants.R_const[camera]
    B = constants.B_const[camera]
    F = constants.F_const[camera]
    O = constants.O_const[camera]
    # J0 = J0[camera]
    # J1 = J1[camera]
    
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/constants.TRefl)-F))
    # # Pseudo radiance of the atmosphere
    r2 = ((1 - constants.Tau)/(Emiss * constants.Tau)) * (R/(np.exp(B/constants.TAtm)-F)) 
    # # Pseudo radiance of the external optics
    r3 = ((1-constants.TransmissionExtOptics) / (Emiss * constants.Tau * constants.TransmissionExtOptics)) * (R/(np.exp(B/constants.TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    return ((R/(np.exp(B/temp)-F)) + O + K2) / K1

## Mouse Callback
selected_pixels = []
rectangle_start = None
rectangle_end = None
update_plot = False
def mouse_callback(event, x, y, flags, param):
    global selected_pixels, rectangle_start, rectangle_end, update_plot

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)

        if selected_pixel not in selected_pixels:
            selected_pixels.append(selected_pixel)
        update_plot = True
        
    elif event == cv2.EVENT_MBUTTONDOWN:
        rectangle_start = (x, y)

    elif event == cv2.EVENT_MBUTTONUP:
        rectangle_end = (x, y)

        pixels_in_rectangle = []
        x1 = min(rectangle_start[0], rectangle_end[0])
        x2 = max(rectangle_start[0], rectangle_end[0])
        y1 = min(rectangle_start[1], rectangle_end[1])
        y2 = max(rectangle_start[1], rectangle_end[1])

        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                pixels_in_rectangle.append((x, y))

        selected_pixels.extend(pixels_in_rectangle)
        update_plot = True

## Display Frame
dot_colors = np.random.randint(0, 255, (3000, 3), dtype=int)
def display_frame(frame, min_value=None, max_value=None, window_name="Frame"):
    if min_value is None or max_value is None:
        min_value = np.min(frame)
        max_value = np.max(frame)
    frame_info_min = f"Min: {min_value}"
    frame_info_max = f"Max: {max_value}"
    # Normalize the frame values between 0 and 1
    frame_normalized = (frame - min_value) / (max_value - min_value)

    # Apply the hot colormap
    frame_colormap = cm.viridis(frame_normalized)
    # Convert the colormap to 8-bit unsigned integer format
    frame_uint8 = (frame_colormap * 255).astype(np.uint8)
    # Convert the colormap to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

    cv2.putText(frame_bgr, frame_info_min,
                (frame_bgr.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame_bgr, frame_info_max,
                (frame_bgr.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for i, pixel in enumerate(selected_pixels):
        x, y = pixel
        pixel_info = f"Pixel: ({x}, {y})"
        # cv2.putText(frame, pixel_info, (10, frame.shape[0] - 40 - i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, dot_colors[i], 2)
        if len(selected_pixels) < dot_colors.shape[0]:
            color = tuple(dot_colors[i].astype(int).tolist())
        else:
            color = tuple(dot_colors[-1].astype(int).tolist())
        cv2.circle(frame_bgr, pixel, 2, color, -1)

    if rectangle_start and rectangle_end:
        cv2.rectangle(frame_bgr, rectangle_start,
                      rectangle_end, (0, 255, 0), 2)

    cv2.imshow(window_name, frame_bgr)

## Object Mask from Video
def get_object_mask_temp(frames_temp, ambient_temp = 295.372, relative_threshold = 5, largest_only=True):
    
    frames_temp = np.array(frames_temp)
    frames_mask = frames_temp > (ambient_temp+relative_threshold)
    frames_mask = np.any(frames_mask, axis=0)

    if largest_only:
        # Get the largest connected component in the tree form
        contours, hierarchy = cv2.findContours(frames_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a blank image of the same size as the original mask
        largest_mask = np.zeros_like(frames_mask, dtype=np.uint8)

        # Draw the largest contour on the blank image
        cv2.drawContours(largest_mask, [largest_contour], -1, (1), thickness=cv2.FILLED)

        # Find the child contours of the largest contour
        child_contours = [contours[i] for i, h in enumerate(hierarchy[0]) if h[3] == 0]

        # Draw the child contours on largest_mask with zeros
        cv2.drawContours(largest_mask, child_contours, -1, (0), thickness=cv2.FILLED)

        return largest_mask
    return frames_mask

## Get a perspective Camera
def get_perspective_camera(img_size, HFOV, VFOV, R, T, device):
    fx = (img_size[1]/2)/np.tan(np.deg2rad(HFOV/2))
    fy = (img_size[0]/2)/np.tan(np.deg2rad(VFOV/2))

    focal_length = torch.tensor([[fx, fy] for i in range(R.shape[0])], device=device, dtype=torch.float32) # Width, Height
    principal_point = torch.tensor([[img_size[1]/2, img_size[0]/2] for i in range(R.shape[0])], device=device) # Width, Height

    # The definition of image_size is now (height, width)
    torch_image_size = torch.tensor([[img_size[0], img_size[1]] for i in range(R.shape[0])], device=device, dtype=torch.float32) # Height, Width
    cameras = PerspectiveCameras(device=device, focal_length=focal_length, principal_point=principal_point, R=R, T=T, image_size=torch_image_size, in_ndc=False)
    return cameras

## Project Meshvals to Image
def project_meshvals_to_img(u_real, faces, fragments_nviews, img_size, no_hatfunc=False):
    global AMBIENT_TEMP, OBJECT_TEMP

    u_real_pixels = []
    for i in range(fragments_nviews.pix_to_face.shape[0]):
        frag_pix_to_face = fragments_nviews.pix_to_face[i]
        valid_ind = torch.where(frag_pix_to_face[..., 0] != -1)
        frag_pix_to_bary = clip_barycentric_coordinates(
            fragments_nviews.bary_coords[i])

        # Face indices get offset by i*faces.shape[0] because we are rendering multiple views
        visible_faces = frag_pix_to_face[valid_ind] - (i * faces.shape[0])
        valid_face_vert = faces[visible_faces].reshape(-1, 3)
        valid_bary_coord = frag_pix_to_bary[valid_ind].reshape(-1, 3)

        u_real_pixels_t = []
        for k in range(u_real.shape[0]):
            tmp_heatval = u_real[k]
            face_vert_heatval = tmp_heatval[valid_face_vert]
            if no_hatfunc:
                valid_hatfaces = torch.count_nonzero(
                    face_vert_heatval != 0, dim=-1)
                valid_hatfaces = (valid_hatfaces > 2).type(torch.int)
                interp_valid_heatval = torch.sum(
                    face_vert_heatval * valid_bary_coord, dim=-1) * valid_hatfaces
            else:
                interp_valid_heatval = torch.sum(
                    face_vert_heatval * valid_bary_coord, dim=-1)
            patch_dist = torch.full((img_size, img_size), OBJECT_TEMP)
            patch_dist[valid_ind] = interp_valid_heatval
            u_real_pixels_t.append(patch_dist)
        u_real_pixels.append(torch.stack(u_real_pixels_t, dim=0))

    return torch.stack(u_real_pixels, dim=0)

## Render Vertices as Image
def render_vertices_as_image(u_vals, pixel_coords, pix_image_num, img_size, fill_value=None, init_image=None, renderNviews=None):
    u_vals_pixels = []
    if fill_value is None:
        min_vals = u_vals.min()
    else:
        min_vals = fill_value
    if renderNviews is None:
        renderNviews = torch.count_nonzero(torch.unique(pix_image_num) != -1).cpu().numpy()
    for i in range(renderNviews):
        if init_image is not None:
            u_vals_img = init_image
        else:
            u_vals_img = torch.full((u_vals.shape[0], img_size[0], img_size[1]), min_vals, dtype=u_vals.dtype, device=u_vals.device)
        valid_verts = torch.where(pix_image_num[:,i] == i)
        img_coords = pixel_coords[valid_verts]
        u_vals_img[:, img_coords[:, 0], img_coords[:, 1]] = u_vals[:, valid_verts[0]]
        u_vals_pixels.append(u_vals_img)
    return torch.stack(u_vals_pixels, dim=0)
    
## Get vertices screen coordinates
def get_verts_screen_coords_for_views(fragments_nviews, verts, faces, cameras, img_size):
    num_views = fragments_nviews.pix_to_face.shape[0]
    image_view_points = cameras.transform_points_screen(
        verts, image_size=(img_size, img_size))
    if num_views == 1:
        image_view_points = image_view_points.unsqueeze(0)
    image_view_points = image_view_points[..., :2]

    for i in range(image_view_points.shape[0]):
        mask = torch.zeros(verts.shape[0], dtype=torch.bool)
        valid_pix = torch.where(fragments_nviews.pix_to_face[i, ..., 0] != -1)

        pix_faces = fragments_nviews.pix_to_face[i][valid_pix] - \
            faces.shape[0]*i
        # print("Pix faces", pix_faces.min(), pix_faces.max())

        faces_verts = faces[pix_faces]
        visible_verts_unq = torch.unique(faces_verts)
        mask[visible_verts_unq] = True
        image_view_points[i, ~mask] = -1

    return image_view_points

## Interpolate Heat Image
def interp_vals_for_griddata_nviews(u_vals, verts_screen_pts, img_size, view_masks):
    
    num_views = view_masks.shape[0]
    u_vals_pixels = []
    for i in range(num_views):
        interp_pts = np.stack(np.where(view_masks[i]), axis=-1)
        verts_screen_pts_view = verts_screen_pts[i]
        verts_inview = np.where(verts_screen_pts_view[:, 0] != -1)
        verts_screen_pts_view = verts_screen_pts_view[verts_inview]
        u_vals_view = u_vals[:, verts_inview].squeeze(1)
        # print(verts_screen_pts_view.shape, u_vals_view.shape, )
        u_vals_timgs = []
        for j in range(u_vals_view.shape[0]):
            # print(np.unique(u_vals_view[j]))
            interp_vals = sp.interpolate.griddata(verts_screen_pts_view, u_vals_view[j], (
                interp_pts[:, 1], interp_pts[:, 0]), method='cubic', fill_value=OBJECT_TEMP)
            uval_img = np.full((img_size, img_size), OBJECT_TEMP)
            uval_img[interp_pts[:, 0], interp_pts[:, 1]] = interp_vals
            # print(uval_img.shape)
            u_vals_timgs.append(uval_img)
        u_vals_pixels.append(u_vals_timgs)
    return np.array(u_vals_pixels)

## Get boundary mesh for Tet
def get_boundary_for_tetmesh(tet_mesh_faces):
    boundary_triangles = []

    # Sort test_mesh_faces so that (i,j,k) and (j,i,k) are equivalent
    tet_mesh_faces = np.sort(tet_mesh_faces, axis=-1)
    tet_mesh_triangles = np.concatenate([tet_mesh_faces[:,[0,1,2]], tet_mesh_faces[:,[0,1,3]], tet_mesh_faces[:,[1,2,3]], tet_mesh_faces[:,[0,2,3]]], axis=0)

    # Step 1: Identify surface triangles
    for tet in tet_mesh_faces:
        for face in [[tet[0], tet[1], tet[2]], [tet[0], tet[1], tet[3]], [tet[1], tet[2], tet[3]], [tet[0], tet[2], tet[3]]]:
            # Check if the face has only one adjacent tetrahedron
            if np.sum(np.all(tet_mesh_triangles == np.array(face), axis=1)) == 1:
                boundary_triangles.append(face)

    # Step 2: Remove duplicate triangles
    boundary_triangles = np.unique(boundary_triangles, axis=0)

    # Step 3: Create the boundary mesh
    boundary_mesh_vertices = np.unique(boundary_triangles)
    boundary_mesh_faces = []
    for triangle in boundary_triangles:
        face_vertices = []
        for vertex in triangle:
            face_vertices.append(
                np.where(boundary_mesh_vertices == vertex)[0][0])
        boundary_mesh_faces.append(face_vertices)

    return boundary_mesh_vertices, boundary_mesh_faces

## Get angle between edges for every face
def get_face_angles(verts, faces):

    face_angles = []
    for i in range(3):
        edge1 = verts[faces[:, (i+1)%3]] - verts[faces[:, i]]
        edge2 = verts[faces[:, (i+2)%3]] - verts[faces[:, i]]
        edge1 = edge1 / torch.norm(edge1, dim=-1, keepdim=True)
        edge2 = edge2 / torch.norm(edge2, dim=-1, keepdim=True)
        dot_prod = torch.sum(edge1 * edge2, dim=-1)
        angle = torch.acos(dot_prod)
        face_angles.append(angle)
    return torch.stack(face_angles, dim=-1)

## Get the sparse edge matrix which given two vertices returns the edge index
def get_edge_matrix(verts_np, faces_np):

    edge_matrix = np.zeros((verts_np.shape[0], verts_np.shape[0]), dtype=np.int32)
    edge_count = 0
    edge_lengths = []
    for face in faces_np:
        for i in range(3):
            if not edge_matrix[face[i], face[(i+1)%3]]:
                edge_count += 1
                edge_matrix[face[i], face[(i+1)%3]] = edge_count
                edge_matrix[face[(i+1)%3], face[i]] = edge_count
                edge_lengths.append(np.linalg.norm(verts_np[face[i]] - verts_np[face[(i+1)%3]]))

    return edge_matrix, np.array(edge_lengths), edge_count

## Get theta values from edge lengths
def get_theta_from_edge_lengths(face_edge_idx, edge_lengths):

    face_edge_lengths = edge_lengths[face_edge_idx]
    a = face_edge_lengths[:, 0]
    b = face_edge_lengths[:, 1]
    c = face_edge_lengths[:, 2]
    theta_gamma = torch.acos((b**2 + c**2 - a**2) / (2*b*c))
    theta_alpha = torch.acos((a**2 + c**2 - b**2) / (2*a*c))
    theta_beta = np.pi - theta_alpha - theta_gamma

    return torch.stack([theta_alpha, theta_beta, theta_gamma], dim=-1)

## Get area of faces from edge lengths
def get_area_from_edge_lengths(face_edge_idx, edge_lengths):

    face_edge_lengths = edge_lengths[face_edge_idx]
    a = face_edge_lengths[:, 0]
    b = face_edge_lengths[:, 1]
    c = face_edge_lengths[:, 2]
    s = (a + b + c) / 2
    area = torch.sqrt((s * (s-a) * (s-b) * (s-c)))
    return area

def get_area_from_face_edge_lengths(face_edge_lengths):

    a = face_edge_lengths[:, 0]
    b = face_edge_lengths[:, 1]
    c = face_edge_lengths[:, 2]
    s = (a + b + c) / 2
    area = torch.sqrt((s * (s-a) * (s-b) * (s-c)))
    return area

## Compute face normals (pytorch)
def compute_face_normals_torch(verts, faces, normalize=False):
    verts_faces = verts[faces]

    edge1 = verts_faces[:, 1] - verts_faces[:, 0]
    edge2 = verts_faces[:, 2] - verts_faces[:, 0]
    
    cross_prod = torch.cross(edge1, edge2, dim=-1)

    face_normals = cross_prod
    neg_idx = torch.where(face_normals[:, -1] < 0)
    face_normals[neg_idx] = -face_normals[neg_idx]
    if normalize:
        face_normals = face_normals / torch.norm(face_normals, dim=-1, keepdim=True)
        return face_normals
    else:
        return face_normals

## Compute per vertex normals (pytorch)
def compute_vertex_normals_torch(verts, faces, camera_loc=torch.tensor([0,0,100.0], device='cuda')):
    
    face_normals = compute_face_normals_torch(verts, faces)
    # camera_dir = camera_loc[None, :] - verts

    # # Make sure the face normals point in the same direction
    # pos_norms = torch.sum(cross_prod * camera_dir[faces[:, 0]], dim=-1, keepdim=True)
    # neg_norms = torch.sum(-cross_prod * camera_dir[faces[:,0]], dim=-1, keepdim=True)
    # face_normals = torch.where(pos_norms >= neg_norms, cross_prod, -cross_prod)

    # face_normals = torch.where(torch.sum(cross_prod * verts_faces[:, 0], dim=-1, keepdim=True) >= 0, cross_prod, -cross_prod)

    # Compute the vertex normals by averaging the face normals based on triangle area
    face_normals_stacked = torch.stack([face_normals, face_normals, face_normals], dim=1)

    zero_idx = torch.zeros_like(faces)
    idx = torch.stack([faces, zero_idx])

    normals_x = torch.sparse_coo_tensor(idx.reshape(2,-1), face_normals_stacked[..., 0].reshape(-1), size=(verts.shape[0], 1)).coalesce().to_dense()
    normals_y = torch.sparse_coo_tensor(idx.reshape(2,-1), face_normals_stacked[..., 1].reshape(-1), size=(verts.shape[0], 1)).coalesce().to_dense()
    normals_z = torch.sparse_coo_tensor(idx.reshape(2,-1), face_normals_stacked[..., 2].reshape(-1), size=(verts.shape[0], 1)).coalesce().to_dense()

    normals = torch.cat([normals_x, normals_y, normals_z], dim=-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    return normals


## Compute per vertex normals (numpy)
def compute_vertex_normals_numpy(verts, faces, camera_loc=np.array([0,0,100.0])):
    
    # verts_faces = verts[faces]

    # edge1 = verts_faces[:, 1] - verts_faces[:, 0]
    # edge2 = verts_faces[:, 2] - verts_faces[:, 0]
    
    # cross_prod = np.cross(edge1, edge2)


    # # Make sure the face normals point in the same direction
    # # face_normals = np.where(np.sum(cross_prod * normal_side[None,:], axis=-1, keepdims=True) >= 0, -cross_prod, cross_prod)
    # face_normals = cross_prod
    with torch.no_grad():
        face_normals = compute_face_normals_torch(torch.from_numpy(verts).cuda().float(), torch.from_numpy(faces).cuda().long())
        face_normals = face_normals.cpu().numpy()

    # Compute the vertex normals by averaging the face normals based on triangle area
    face_normals_stacked = np.stack([face_normals, face_normals, face_normals], axis=0)

    normals_x = np.bincount(faces.reshape(-1), face_normals_stacked[..., 0].reshape(-1), minlength=verts.shape[0])
    normals_y = np.bincount(faces.reshape(-1), face_normals_stacked[..., 1].reshape(-1), minlength=verts.shape[0])
    normals_z = np.bincount(faces.reshape(-1), face_normals_stacked[..., 2].reshape(-1), minlength=verts.shape[0])

    normals = np.stack([normals_x, normals_y, normals_z], axis=-1)
    # camera_dir = camera_loc[None, :] - verts

    # pos_sum = np.sum(normals * camera_dir, axis=-1)
    # neg_sum = np.sum(-normals * camera_dir, axis=-1)

    # flip_locs = np.where(pos_sum < neg_sum)
    # normals[flip_locs] = -normals[flip_locs]

    # normals = np.where(np.sum(normals * camera_dir, axis=-1, keepdims=True) >= 0, normals, -normals)
    # print(np.unique(np.linalg.norm(normals, axis=-1)))
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    return normals


## Save experiemnnts log
def save_experiments_config(experiments_file, filename, exp_name, **kwargs):
    experiments_log = pd.read_csv(os.path.join(workspace_path, f'exp-logs/{experiments_file}.csv'))

    experiments_log = pd.read_csv(os.path.join(workspace_path, f'exp-logs/{experiments_file}.csv'))
    experiment = {}
    experiment['Filename'] = filename
    experiment['Exp-name'] = exp_name

    local_dict = kwargs.get('local_dict', {})
    for k, v in local_dict.items():
        if type(v) in [int, float, bool] and not k.startswith('__'):
            experiment[k] = v

    for k, v in kwargs.get('params_dict', {}).items():
        experiment[k] = v
    for k, v in kwargs.get('Sopti_params_dict', {}).items():
        experiment['Sopti-'+k] = v

    for k,v in kwargs.get('Lopti_params_dict', {}).items():
        experiment['Lopti-'+k] = v
            
    # Append dict to the CSV table
    experiments_log = experiments_log.append(experiment, ignore_index=True)
    experiments_log.to_csv(os.path.join(workspace_path, f'exp-logs/{experiments_file}.csv'), index=False)


def quaternion_to_rotation_matrix(quaternion):
    '''
    Convert quaternion to rotation matrix
    quaternion: [w, x, y, z]
    '''
    r = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    return r.as_matrix()

def extract_colmap_pose_from_txt(images_txt):
    '''
    Extracts the pose information from the colmap images.txt file
    pose_dict = {NAME: {"rotation": R, "translation": t, "camera_id": IMAGE_ID}}
    '''
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)

    pose_dict = {}
    with open(images_txt, 'r') as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):
            line1 = lines[i].strip().split()
            line2 = lines[i+1].strip().split()
            name = line1[-1]
            # print(name, line1)
            R = quaternion_to_rotation_matrix([float(line1[1]), float(line1[2]), float(line1[3]), float(line1[4])])
            t = np.array([float(line1[5]), float(line1[6]), float(line1[7])])
            pose_dict[name] = {"rotation": R, "translation": t, "camera_id": int(line1[-2])}
    return pose_dict

def save_mesh_asobj(verts, faces, filename):
    # Check if the mesh is triangular or tetrachedral
    if faces.shape[1] == 3:
        with open(filename, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    else:
        with open(filename, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")


## Increase image border
def increase_image_border(images, border_width=1, bg_value=100):
    
    mask_img = np.zeros_like(images, dtype=bool)
    mask_img[images != bg_value] = True
    dimg = images.copy()
    dimg[dimg == bg_value] = np.NaN
    xdiff = np.diff(dimg, axis=-1)
    ydiff = np.diff(dimg, axis=-2)
    xdiff[np.isnan(xdiff)] = -0.01
    ydiff[np.isnan(ydiff)] = -0.01
    xdiff = np.pad(xdiff, ((0,0), (0,0), (1,0)), mode='constant', constant_values=0)
    ydiff = np.pad(ydiff, ((0,0), (1,0), (0,0)), mode='constant', constant_values=0)

    # blur xdiff and ydiff
    xdiff = skimage.filters.gaussian(xdiff, sigma=5.8, preserve_range=True, truncate=2.0, cval=0.0)
    ydiff = skimage.filters.gaussian(ydiff, sigma=5.8, preserve_range=True, truncate=2.0, cval=0.0)

    # print(xdiff.shape, ydiff.shape, mask_img.shape)
    for k in range(border_width):
        s1 = np.roll(mask_img, 1, axis=-1) & ~mask_img
        s2 = np.roll(mask_img, -1, axis=-1) & ~mask_img
        s3 = np.roll(mask_img, 1, axis=-2) & ~mask_img
        s4 = np.roll(mask_img, -1, axis=-2) & ~mask_img
        
        # Roll the diff image and add it to the original image
        xdiff[np.where(s1)] = np.roll(xdiff, 1, axis=-1)[np.where(s1)]
        xdiff[np.where(s2)] = np.roll(xdiff, -1, axis=-1)[np.where(s2)]
        
        ydiff[np.where(s3)] = np.roll(ydiff, 1, axis=-2)[np.where(s3)]
        ydiff[np.where(s4)] = np.roll(ydiff, -1, axis=-2)[np.where(s4)]

        images[np.where(s1)] = np.roll(images, 1, axis=-1)[np.where(s1)] - xdiff[np.where(s1)]
        images[np.where(s2)] = np.roll(images, -1, axis=-1)[np.where(s2)] - xdiff[np.where(s2)]
        images[np.where(s3)] = np.roll(images, 1, axis=-2)[np.where(s3)] - ydiff[np.where(s3)]
        images[np.where(s4)] = np.roll(images, -1, axis=-2)[np.where(s4)] - ydiff[np.where(s4)]
        
        mask_img[np.where(s1)] = 1
        mask_img[np.where(s2)] = 1
        mask_img[np.where(s3)] = 1
        mask_img[np.where(s4)] = 1

    images[images == bg_value] = 0.0
    return images