from ripl.config import TrainConfig, parse_config


def test_yaml_config_and_cli_override():
    config = parse_config(
        TrainConfig,
        ["--config", "configs/smoke.yaml", "--total-iters", "12"],
    )
    assert config.total_iters == 12
    assert config.unet_dims == (32, 64, 128)
    assert config.demo_path.endswith("trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")
