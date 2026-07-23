import trimesh
import numpy as np
import copy

def load_and_transform_mesh(mesh_path):
    mesh=trimesh.load(mesh_path,process=True)
    vertices=np.asarray(mesh.vertices)
    bb_min,bb_max=np.amin(vertices,axis=0),np.amax(vertices,axis=0)
    center=(bb_min+bb_max)/2
    size=bb_max-bb_min
    max_size=np.amax(size)
    vertices=vertices-center #center at (0,0,0)
    vertices=vertices/max_size*2
    vertices[:,1]+=size[1]/max_size #y is up, stand on y=0 floor
    mesh.vertices=copy.deepcopy(vertices)
    return mesh
