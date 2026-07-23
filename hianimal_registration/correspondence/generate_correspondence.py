#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


UV_THRESHOLDS = {
    "1": 0.004,
    "2": 0.004,
    "3": 0.004,
    "4": 0.0045,
    "5": 0.0045,
    "6": 0.003,
    "7": 0.002,
}


def load_uv_template(path: Path):
    vertices = []
    uvs = []
    vertex_to_uvs = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(value) for value in line.split()[1:3]])
            elif line.startswith("f "):
                for item in line.split()[1:]:
                    fields = item.split("/")
                    vertex_index = int(fields[0]) - 1
                    if len(fields) > 1 and fields[1]:
                        uv_index = int(fields[1]) - 1
                        vertex_to_uvs.setdefault(vertex_index, set()).add(
                            tuple(uvs[uv_index])
                        )

    unique_indices = []
    unique_uvs = []
    for vertex_index, assigned_uvs in vertex_to_uvs.items():
        if len(assigned_uvs) == 1:
            unique_indices.append(vertex_index)
            unique_uvs.append(next(iter(assigned_uvs)))

    vertices = np.asarray(vertices)
    unique_indices = np.asarray(unique_indices, dtype=np.int64)
    return vertices, unique_indices, np.asarray(unique_uvs)


def load_vertex_groups(path: Path):
    groups = {}
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) == 2:
                vertex_index = int(parts[0].split()[1])
                groups[vertex_index] = parts[1].strip().split()[-1].strip("[]'")
    return groups


def generate(target_obj: Path, target_npz: Path, output_dir: Path):
    asset_dir = Path(__file__).resolve().parent / "assets"
    _, template_indices, template_uvs = load_uv_template(
        asset_dir / "uv_template.obj"
    )
    vertex_groups = load_vertex_groups(asset_dir / "vertex_groups.txt")
    registration_mapping = np.load(asset_dir / "uv_to_registration.npy")

    target_mesh = trimesh.load(target_obj, process=False)
    if isinstance(target_mesh, trimesh.Scene):
        target_mesh = trimesh.util.concatenate(tuple(target_mesh.geometry.values()))
    target_vertices = np.asarray(target_mesh.vertices)
    target_uvs = np.load(target_npz)["uvs"]
    if target_uvs.shape != (target_vertices.shape[0], 2):
        raise ValueError("The UV array must contain one UV coordinate per mesh vertex")

    source_matches = []
    target_matches = []
    for group in map(str, range(1, 8)):
        mask = np.asarray(
            [vertex_groups.get(int(index)) == group for index in template_indices]
        )
        group_indices = template_indices[mask]
        group_uvs = template_uvs[mask]
        if group_indices.size == 0:
            continue

        nearest_target = cKDTree(target_uvs).query(group_uvs)[1]
        uv_distance = np.linalg.norm(
            group_uvs - target_uvs[nearest_target], axis=1
        )
        valid = uv_distance < UV_THRESHOLDS[group]

        source_matches.extend(registration_mapping[group_indices[valid]])
        target_matches.extend(nearest_target[valid])
        print(f"group {group}: {int(valid.sum())} matches")

    correspondence = np.full(target_vertices.shape[0], -1, dtype=np.int32)
    for source_index, target_index in zip(source_matches, target_matches):
        correspondence[target_index] = source_index

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "correspondence.npy", correspondence)
    print(f"saved {int((correspondence >= 0).sum())} correspondences")
    print(f"correspondence: {output_dir / 'correspondence.npy'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-obj", type=Path, required=True)
    parser.add_argument("--target-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.target_obj, args.target_npz, args.output_dir)


if __name__ == "__main__":
    main()
