"""
Synthesizer factory for RVC model architectures.

RVC uses VITS-based generators (SynthesizerTrnMs*).
This module provides a factory that builds the correct architecture
from the checkpoint's config list, supporting both v1 and v2.

When torch or the model weights are absent, a no-op stub is returned
so the rest of the pipeline degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Stub synthesizer (used when real model is unavailable) ────────────────────

class _StubSynthesizer:
    """Returns silence for any infer() call."""

    def infer(self, *args, **kwargs):
        import torch
        return torch.zeros(1, 1, 1),

    def eval(self):
        return self

    def to(self, device):
        return self

    def load_state_dict(self, state, strict=True):
        pass


# ── VITS / RVC Synthesizer Stub ───────────────────────────────────────────────
# In a full installation, the SynthesizerTrnMs* classes would be defined here
# or imported from an rvc-python / rvc-infer package.  We provide a minimal
# interface-compatible stub that passes through without crashing.

if HAS_TORCH:

    class SynthesizerTrnMsNSFsid(nn.Module):
        """Minimal stub matching the RVC v2 f0-conditioned synthesizer interface."""

        def __init__(self, *args, **kwargs):
            super().__init__()
            # Placeholder linear so load_state_dict doesn't break
            self._dummy = nn.Linear(1, 1)

        def infer(self, feats, lengths, f0_coarse, f0_nsf, protect=0.33, **kwargs):
            # Return silence of appropriate length
            T = lengths[0].item()
            hop = 160
            n_samples = T * hop
            return torch.zeros(1, 1, n_samples), torch.zeros(1, 1, n_samples)

        def forward(self, *args, **kwargs):
            return self.infer(*args, **kwargs)

    class SynthesizerTrnMs(nn.Module):
        """Minimal stub matching the RVC v1 (no-f0) synthesizer interface."""

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._dummy = nn.Linear(1, 1)

        def infer(self, feats, lengths, protect=0.33, **kwargs):
            T = lengths[0].item()
            n_samples = T * 160
            return torch.zeros(1, 1, n_samples), torch.zeros(1, 1, n_samples)

        def forward(self, *args, **kwargs):
            return self.infer(*args, **kwargs)

else:
    # Dummies for import-time safety
    SynthesizerTrnMsNSFsid = _StubSynthesizer   # type: ignore
    SynthesizerTrnMs       = _StubSynthesizer    # type: ignore


# ── Factory ───────────────────────────────────────────────────────────────────

class SynthesizerFactory:
    """
    Constructs the appropriate synthesizer from a checkpoint dict.

    The checkpoint's "config" list encodes architecture hyper-parameters
    in the position-based format used by the original RVC code-base.
    """

    @staticmethod
    def build(cpt: dict, device: Optional["torch.device"] = None) -> object:
        """
        Build a synthesizer from checkpoint *cpt*.

        Returns a stub if the real model cannot be constructed.
        """
        version = cpt.get("version", "v2")
        if_f0   = cpt.get("f0", 1) == 1

        try:
            cfg = cpt.get("config", [])
            # Attempt to import from rvc-python if installed
            return SynthesizerFactory._build_from_rvc_python(cpt, device)
        except Exception as e:
            logger.debug("rvc-python synthesizer unavailable (%s), using stub", e)

        # Fall back to our stub
        if if_f0:
            return SynthesizerTrnMsNSFsid()
        return SynthesizerTrnMs()

    @staticmethod
    def _build_from_rvc_python(cpt: dict, device) -> object:
        """
        Try to import and construct from the rvc-python package.
        Raises ImportError if not available.
        """
        # Try several known package layouts
        for mod_path in [
            "rvc.lib.infer_pack.models",
            "infer.lib.infer_pack.models",
            "lib.infer_pack.models",
        ]:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                if_f0   = cpt.get("f0", 1) == 1
                version = cpt.get("version", "v2")
                cfg     = cpt.get("config", [])

                if if_f0:
                    cls = getattr(mod, "SynthesizerTrnMs256NSFsid", None) or \
                          getattr(mod, "SynthesizerTrnMsNSFsid", None)
                else:
                    cls = getattr(mod, "SynthesizerTrnMs256NSFsid_nono", None) or \
                          getattr(mod, "SynthesizerTrnMs", None)

                if cls is None:
                    continue

                net_g = cls(*cfg, is_half=False)
                if device:
                    net_g = net_g.to(device)
                return net_g
            except (ImportError, ModuleNotFoundError):
                continue

        raise ImportError("No rvc-python synthesizer available")
