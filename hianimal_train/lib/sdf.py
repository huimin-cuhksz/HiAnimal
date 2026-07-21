import numpy as np


def create_grid(resolution, b_min, b_max):
    coordinates = np.mgrid[:resolution, :resolution, :resolution].reshape(3, -1)
    transform = np.eye(4)
    transform[0, 0] = (b_max[0] - b_min[0]) / resolution
    transform[1, 1] = (b_max[1] - b_min[1]) / resolution
    transform[2, 2] = (b_max[2] - b_min[2]) / resolution
    transform[:3, 3] = b_min
    coordinates = transform[:3, :3] @ coordinates + transform[:3, 3:4]
    return coordinates.reshape(3, resolution, resolution, resolution), transform


def batch_eval(points, evaluate, batch_size=10000):
    values = np.zeros(points.shape[1])
    for start in range(0, points.shape[1], batch_size):
        end = min(start + batch_size, points.shape[1])
        values[start:end] = evaluate(points[:, start:end])
    return values


def eval_grid_octree(coordinates, evaluate, init_resolution=64, threshold=0.01):
    resolution = coordinates.shape[1]
    flattened = coordinates.reshape(3, -1)
    values = np.zeros((resolution, resolution, resolution))
    dirty = np.ones(values.shape, dtype=bool)
    grid_mask = np.zeros(values.shape, dtype=bool)
    step = resolution // init_resolution

    while step > 0:
        grid_mask[::step, ::step, ::step] = True
        test_mask = grid_mask & dirty
        values[test_mask] = batch_eval(flattened[:, test_mask.reshape(-1)], evaluate)
        dirty[test_mask] = False
        if step <= 1:
            break

        for x in range(0, resolution - step, step):
            for y in range(0, resolution - step, step):
                for z in range(0, resolution - step, step):
                    center = (x + step // 2, y + step // 2, z + step // 2)
                    if not dirty[center]:
                        continue
                    corners = values[
                        np.ix_((x, x + step), (y, y + step), (z, z + step))
                    ]
                    if corners.max() - corners.min() < threshold:
                        value = (corners.max() + corners.min()) / 2
                        values[x : x + step, y : y + step, z : z + step] = value
                        dirty[x : x + step, y : y + step, z : z + step] = False
        step //= 2

    return values
