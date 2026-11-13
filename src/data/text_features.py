from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


class HashTextEncoder:
    """Deterministic stand-in for a FROZEN text encoder, used by the toy path.

    [IMPORTANT] This is NOT a semantic encoder. It maps token strings to fixed
    pseudo-random vectors via a hash, so that DIFFERENT prompts produce
    DIFFERENT features and the same prompt always produces the SAME features.
    It exists solely to verify that a text signal can travel through the
    trainable branch and change the output. Any claim about language
    understanding must come from the real Qwen3-4B path, not from this file.
    """

    def __init__(self, dim: int, num_tokens: int = 8, seed: int = 20261107, vocab: int = 8192):
        self.dim, self.num_tokens, self.vocab = dim, num_tokens, vocab
        g = np.random.default_rng(seed)
        self.table = torch.from_numpy(
            g.normal(0, 1 / np.sqrt(dim), (vocab, dim)).astype(np.float32)
        )
        self.pad = torch.zeros(dim, dtype=torch.float32)

    @staticmethod
    def _tokens(text: str):
        clean = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
        return [w for w in clean.split() if w]

    @staticmethod
    def _idx(word: str, vocab: int) -> int:
        return int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % vocab

    def encode(self, texts) -> torch.Tensor:
        """list[str] -> [B, L, D] float32 (no grad)."""
        rows = []
        for text in texts:
            toks = self._tokens(text)[: self.num_tokens]
            vecs = [self.table[self._idx(w, self.vocab)] for w in toks]
            while len(vecs) < self.num_tokens:
                vecs.append(self.pad)
            rows.append(torch.stack(vecs))
        return torch.stack(rows)


class OfflineTextCache:
    """Disk cache of frozen text-encoder outputs.

    On a 12 GB card this is not an optimisation, it is a requirement: the
    Qwen3-4B encoder (~8 GB BF16) is run once on CPU/offline and NEVER loaded
    into VRAM during training. At inference, free user text must be encoded
    on the fly (see src/backends/flux2_klein.py for the offload strategy).
    """

    def __init__(self, root: Path, encoder, namespace: str | None = None):
        # The cache key is content-only, so it MUST be namespaced by encoder.
        # Otherwise a 64-dim toy embedding gets loaded into a 256-dim (or
        # Qwen3 2560-dim) projector and fails with a shape mismatch.
        if namespace is None:
            d = getattr(encoder, "dim", None)
            n = getattr(encoder, "num_tokens", None)
            namespace = f"dim{d}_t{n}" if d is not None else "default"
        self.namespace = namespace
        self.dir = Path(root) / "data" / "text_cache" / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]

    def _load(self, p: Path):
        """Load one cached embedding, self-healing a truncated writer.

        2026-11-13: two training processes building the SAME cache for the
        first time used to race -- one read a half-written .pt and died with
        "PytorchStreamReader failed finding central directory". Recompute
        instead of crashing a 1200-step run; the encoder is deterministic.
        """
        try:
            return torch.load(p, weights_only=True)
        except Exception:                                   # noqa: BLE001
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def _save(self, p: Path, tensor) -> None:
        """Atomic publish: readers only ever see a COMPLETE .pt file."""
        import os
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp")
        torch.save(tensor, tmp)
        os.replace(tmp, p)

    def get(self, texts):
        """Returns [B,L,D]; encodes and caches any missing/corrupt prompts."""
        out, missing, missing_idx = [], [], []
        for i, t in enumerate(texts):
            p = self.dir / f"{self._key(t)}.pt"
            v = self._load(p) if p.exists() else None
            if v is not None:
                out.append(v)
            else:
                out.append(None)
                missing.append(t)
                missing_idx.append(i)
        if missing:
            enc = self.encoder.encode(missing)
            for j, t in enumerate(missing):
                self._save(self.dir / f"{self._key(t)}.pt", enc[j])
                out[missing_idx[j]] = enc[j]
        return torch.stack(out)
