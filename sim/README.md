# sim

MuJoCo simulation for the OpenArm bimanual setup.

## Model source

The model lives outside this repo in `third_party/openarm_mujoco` (git-ignored, clone it yourself):

```sh
git clone https://github.com/enactic/openarm_mujoco third_party/openarm_mujoco
git -C third_party/openarm_mujoco checkout 36d4a443533478a6a1b6e1d6e0d29b35583afc50
```

| | |
|---|---|
| Upstream | https://github.com/enactic/openarm_mujoco |
| Pinned commit | `36d4a443533478a6a1b6e1d6e0d29b35583afc50` |
| Commit date | 2026-10-16 |
| Subject | Recalculate V1 link7, hand and finger inertias from measured masses (#90) |
| Model file | `third_party/openarm_mujoco/v1/openarm_bimanual.xml` |

Loading `v1/openarm_bimanual.xml` at that commit gives `njnt = 18`, `nu = 16`
(7 arm joints + 2 finger joints per arm; 7 arm actuators + 1 gripper actuator per arm).

## Headless rendering

`MUJOCO_GL=egl` works on this machine; `MUJOCO_GL=osmesa` also works as a fallback.

```sh
MUJOCO_GL=egl uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('third_party/openarm_mujoco/v1/openarm_bimanual.xml'); r=mujoco.Renderer(m); r.close()"
```

Call `Renderer.close()` explicitly — letting the renderer be collected at interpreter
shutdown raises a harmless `EGLError` from its destructor.
