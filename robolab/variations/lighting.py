# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class SphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )


@configclass
class RedSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    red_sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere_red",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(1.0, 0.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
@configclass
class BlueSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    blue_sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere_blue",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(0.0, 0.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
@configclass
class GreenSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    green_sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere_green",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(0.0, 1.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@configclass
class ExtremelyDimSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=50),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

@configclass
class FrontDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    front_directional_light = AssetBaseCfg(
        prim_path="/World/front_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.0, 0.7071, 0.0)),
    )

@configclass
class BehindDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    behind_directional_light = AssetBaseCfg(
        prim_path="/World/behind_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.0, -0.7071, 0.0)),
    )

@configclass
class TopDownDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    top_down_directional_light = AssetBaseCfg(
        prim_path="/World/top_down_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=0.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

@configclass
class LeftDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    left_directional_light = AssetBaseCfg(
        prim_path="/World/left_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, -0.7071, 0.0, 0.0)),
    )

@configclass
class RightDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    right_directional_light = AssetBaseCfg(
        prim_path="/World/right_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
    )


@configclass
class WarmDaylightCfg:
    """Warm window-like key light with a soft neutral fill."""
    key_light = AssetBaseCfg(
        prim_path="/World/warm_daylight_key",
        spawn=sim_utils.DistantLightCfg(intensity=2200, angle=2.0, exposure=1.0, color=(1.0, 0.90, 0.76)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, -0.52, 0.48, 0.0)),
    )
    fill_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/warm_daylight_fill",
        spawn=sim_utils.SphereLightCfg(intensity=1800, color=(0.82, 0.90, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, -0.5, 1.8)),
    )


@configclass
class CoolDaylightCfg:
    """Cool daylight key balanced by a warm interior fill."""
    key_light = AssetBaseCfg(
        prim_path="/World/cool_daylight_key",
        spawn=sim_utils.DistantLightCfg(intensity=2100, angle=2.0, exposure=1.0, color=(0.76, 0.87, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.48, 0.52, 0.0)),
    )
    fill_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/cool_daylight_fill",
        spawn=sim_utils.SphereLightCfg(intensity=1700, color=(1.0, 0.82, 0.62)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.45, 1.6)),
    )


@configclass
class WarmIndoorLightCfg:
    """Soft warm practical lighting suitable for residential rooms."""
    ceiling_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/warm_indoor_ceiling",
        spawn=sim_utils.SphereLightCfg(intensity=3600, color=(1.0, 0.72, 0.46)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, 2.35)),
    )
    fill_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/warm_indoor_fill",
        spawn=sim_utils.SphereLightCfg(intensity=1200, color=(0.72, 0.84, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3, -0.7, 1.3)),
    )


@configclass
class NeutralStudioLightCfg:
    """Balanced key/fill lighting that preserves surface colors."""
    key_light = AssetBaseCfg(
        prim_path="/World/neutral_studio_key",
        spawn=sim_utils.DistantLightCfg(intensity=1800, angle=3.0, exposure=1.0, color=(1.0, 0.95, 0.88)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, -0.60, 0.38, 0.0)),
    )
    fill_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/neutral_studio_fill",
        spawn=sim_utils.SphereLightCfg(intensity=2200, color=(0.82, 0.90, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.1, 0.55, 1.65)),
    )
