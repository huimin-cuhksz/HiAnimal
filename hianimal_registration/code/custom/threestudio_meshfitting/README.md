# threestudio-meshfitting
![mesh-fitting](https://github.com/DSaurus/threestudio-meshfitting/assets/24589363/236bbad3-902b-4212-9521-99166de8e672)

A simple extension of threestudio, which enables using neural representation to fit a 3D mesh. To use it, please install [threestudio](https://github.com/threestudio-project/threestudio) first and then install this extension in threestudio `custom` directory.

# Installation
```
cd custom
git clone https://github.com/DSaurus/threestudio-meshfitting.git
```

# Quick Start
```
python launch.py --config custom/threestudio_meshfitting/configs/mesh-fitting.yaml --train system.geometry.shape_init="mesh:custom/threestudio_meshfitting/assets/a_Bulbasaur.obj" system.guidance.geometry.shape_init="mesh:custom/threestudio_meshfitting/assets/a_Bulbasaur.obj"
```
/usr/bin/python launch.py --config custom/threestudio_meshfitting/configs/deform-fitting.yaml --train system.geometry.shape_init="mesh:3DBiCar/3763/tpose/m.obj"
CUDA_VISIBLE_DEVICES='3' python launch.py --config custom/threestudio_meshfitting/configs/jacobian-fitting.yaml --train system.geometry.shape_init="mesh:3DBiCar/3763/tpose/m_tri.obj"