from ripl.config import TrainConfig, parse_config


def test_yaml_config_and_cli_override():
    config = parse_config(
        TrainConfig,
        ["--config", "configs/smoke.yaml", "--total-iters", "12"],
    )
    assert config.total_iters == 12
    assert config.unet_dims == (32, 64, 128)
    assert config.demo_path.endswith("trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")


def test_spatial_config_preserves_image_location():
    config = parse_config(TrainConfig, ["--config", "configs/pusht_rgb_spatial.yaml"])
    assert config.pool_visual_feature_map is False
    assert config.act_horizon == 8


def test_pushcube_config_uses_official_task_settings():
    config = parse_config(TrainConfig, ["--config", "configs/pushcube_rgb.yaml"])
    assert config.env_id == "PushCube-v1"
    assert config.control_mode == "pd_ee_delta_pos"
    assert config.obs_mode == "rgb"
    assert config.sim_backend == "physx_cpu"
    assert config.max_episode_steps == 100
    assert config.total_iters == 30_000
    assert config.act_horizon == 8
    assert config.pool_visual_feature_map is False


def test_pusht_20pct_config_uses_supported_controller_and_task_tuned_horizon():
    config = parse_config(TrainConfig, ["--config", "configs/pusht_rgb_20pct.yaml"])
    assert config.env_id == "PushT-v1"
    assert config.control_mode == "pd_ee_delta_pos"
    assert config.demo_path.endswith("trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")
    assert config.sim_backend == "physx_cuda"
    assert config.max_episode_steps == 150
    assert config.total_iters == 50_000
    assert config.batch_size == 256
    assert config.act_horizon == 1
    assert config.pool_visual_feature_map is False
