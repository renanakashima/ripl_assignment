import pytest

torch = pytest.importorskip("torch")

from ripl.networks import ConditionalUnet1D, PlainConv


def test_conditional_unet_preserves_action_shape():
    model = ConditionalUnet1D(
        input_dim=4,
        global_cond_dim=32,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32, 64),
        n_groups=8,
    )
    actions = torch.randn(2, 16, 4)
    condition = torch.randn(2, 32)
    prediction = model(actions, torch.tensor([1, 9]), condition)
    assert prediction.shape == actions.shape


def test_visual_encoder_accepts_128_square_rgb():
    encoder = PlainConv(in_channels=3, out_dim=64)
    assert encoder(torch.rand(2, 3, 128, 128)).shape == (2, 64)


def test_visual_encoder_can_preserve_spatial_feature_map():
    encoder = PlainConv(
        in_channels=3,
        out_dim=64,
        image_size=(128, 128),
        pool_feature_map=False,
    )
    assert encoder.fc[0].in_features == 128 * 8 * 8
    assert encoder(torch.rand(2, 3, 128, 128)).shape == (2, 64)
