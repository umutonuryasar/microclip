"""Contrastive objectives compared in this project, plus the config-driven factory."""
from .infonce import infonce_loss
from .siglip import siglip_loss

__all__ = ["build_loss", "infonce_loss", "siglip_loss"]


def build_loss(cfg: dict):
    """Returns callable(img_feats, txt_feats, model) -> scalar, dispatched on
    cfg["loss"]["type"]. The model is passed so each loss can pull the
    learnable temperature (and, for sigmoid, the bias) it needs."""
    kind = cfg["loss"]["type"]
    if kind == "sigmoid":
        return lambda img, txt, model: siglip_loss(img, txt, model.logit_scale, model.logit_bias)
    if kind == "softmax":
        return lambda img, txt, model: infonce_loss(img, txt, model.logit_scale)
    raise ValueError(f"Unknown loss type: {kind}")
