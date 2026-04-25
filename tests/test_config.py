from chimera.config import load_config, scale_config


def test_config_scaling_without_torch_runtime():
    cfg = scale_config(load_config("config.json"), "nano")
    assert cfg["hidden_size"] == 128
    assert cfg["num_hidden_layers"] == 4
    assert cfg["vocab_size"] <= 8192
