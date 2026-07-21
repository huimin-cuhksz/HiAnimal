import torch
import trimesh
from skimage import measure

from .sdf import create_grid, eval_grid_octree


def reconstruct(net, device, calibration, resolution, b_min, b_max):
    coordinates, transform = create_grid(resolution, b_min, b_max)

    def evaluate_occupancy(points):
        samples = torch.from_numpy(points).float().unsqueeze(0).to(device)
        net.query(samples, calibration)
        return net.get_preds()[0, 0].detach().cpu().numpy()

    occupancy = eval_grid_octree(coordinates, evaluate_occupancy)
    vertices, faces, _, _ = measure.marching_cubes_lewiner(occupancy, 0.5)
    vertices = (transform[:3, :3] @ vertices.T + transform[:3, 3:4]).T

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = max(mesh.split(), key=lambda component: component.vertices.shape[0])
    mesh = mesh.subdivide()

    points = torch.from_numpy(mesh.vertices.T).float().unsqueeze(0).to(device)
    net.query(points, calibration)
    uv = net.get_preds()[0, 1:3].detach().cpu().numpy().T
    return mesh.vertices, mesh.faces, uv


def save_obj(path, vertices, faces):
    with open(path, "w", encoding="utf-8") as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]:.4f} {vertex[1]:.4f} {vertex[2]:.4f}\n")
        for face in faces + 1:
            file.write(f"f {face[0]} {face[2]} {face[1]}\n")
