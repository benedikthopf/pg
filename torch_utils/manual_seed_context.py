import torch


class ManualSeed:
    """Context manager: Fixes pytorchs manual seed, then resets it when the context is left"""

    def __init__(self, maual_seed=0):
        self.rng_state = None
        self.manual_seed = 0

    def __enter__(self):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.manual_seed)

    def __exit__(self, exc_type, exc_value, exc_tb):
        torch.random.set_rng_state(self.rng_state)
