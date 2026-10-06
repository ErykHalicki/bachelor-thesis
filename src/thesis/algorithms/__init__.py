def build_algorithm(cfg):
    if cfg.name == "dummy":
        from .dummy import DummyAlgorithm
        return DummyAlgorithm(cfg)
    if cfg.name == "generic_vit_predictor":
        from .generic_vit_predictor import GenericViTPredictor
        return GenericViTPredictor(cfg)
    if cfg.name == "lerobot_policy":
        from .lerobot_policy import LeRobotPolicy
        return LeRobotPolicy(cfg)
    if cfg.name == "latent_decoder":
        from .latent_decoder import LatentPixelDecoder
        return LatentPixelDecoder(cfg)
    if cfg.name == "wam_decoder":
        from .wam_decoder import WAMPixelDecoder
        return WAMPixelDecoder(cfg)
    raise ValueError(f"unknown algorithm '{cfg.name}'")
