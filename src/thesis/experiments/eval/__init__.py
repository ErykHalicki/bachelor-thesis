def build_eval(cfg):
    """Lazy eval-backend factory, mirroring datasets/. Same `backend=` dimension picks
    the matching eval environment, so training and eval always share one world.
    """
    backend = cfg.backend
    if backend == "dummy":
        from .dummy import DummyEval
        return DummyEval(cfg)
    if backend == "offline":
        from .offline import OfflineEval
        return OfflineEval(cfg)
    if backend == "reconstruction":
        from .reconstruction import ReconstructionEval
        return ReconstructionEval(cfg)
    if backend == "robocasa":
        from .robocasa import RoboCasaEval
        return RoboCasaEval(cfg)
    if backend == "lerobot":
        from .lerobot import LeRobotEval
        return LeRobotEval(cfg)
    raise ValueError(
        f"eval backend '{backend}' not installed. "
        f"try: uv pip install 'bachelor-thesis[{backend}]'"
    )
