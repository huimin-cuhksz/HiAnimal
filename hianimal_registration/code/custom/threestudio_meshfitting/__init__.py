import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.0"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )

from .geometry import obj_mesh
from .guidance import mesh_guidance
from .renderer import mesh_renderer,simple_nvdiff,jacobian_nvdiff
from .system import mesh_fitting,deform_fitting_system,jacobian_deform_system
from .data import deform_data, deform_data_smpl, geometry_registration
from .geometry import deform_mesh,jacobian_mesh
