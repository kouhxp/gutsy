"""Model registry: models.json -> lazily loaded engines (one llama.cpp context per model)."""
import json
import threading
from pathlib import Path

from .engine import Calibration, Engine


class Registry:
    def __init__(self, config_path, n_threads=None, backend_factory=None):
        self.path = Path(config_path)
        cfg = json.loads(self.path.read_text())
        self.cfg = cfg
        self.models = cfg["models"]
        self.default = cfg.get("default") or next(iter(self.models))
        if self.default not in self.models:
            raise ValueError(f"default model {self.default!r} is not in 'models'")
        self.n_threads = n_threads or cfg.get("n_threads")
        self.engines, self.lock = {}, threading.Lock()
        self.backend_factory = backend_factory or self._llamacpp

    def _resolve(self, p):
        p = Path(p)
        return p if p.is_absolute() else (self.path.parent / p)

    def _llamacpp(self, spec):
        from .backend_llamacpp import LlamaCppBackend
        return LlamaCppBackend(self._resolve(spec["gguf"]), n_ctx=spec.get("n_ctx", 8192),
                               n_threads=self.n_threads,
                               n_gpu_layers=self.cfg.get("n_gpu_layers", 0))

    def names(self):
        return list(self.models)

    def get(self, name=None):
        name = name or self.default
        if name not in self.models:
            raise KeyError(name)
        with self.lock:
            if name not in self.engines:
                spec = self.models[name]
                cal = spec.get("calibration")
                engine = Engine(self.backend_factory(spec),
                                Calibration.load(self._resolve(cal) if cal else None), name=name,
                                cache_entries=self.cfg.get("cache_entries", 4),
                                cache_bytes=int(self.cfg.get("cache_mb", 512)) * 2**20,
                                max_options=spec.get("max_options"),
                                reject_slot=bool(spec.get("reject_slot", False)),
                                shortlist=bool(spec.get("shortlist", True)))
                engine.self_check(tol=float(spec.get("cache_tol", 1e-3)))
                self.engines[name] = engine
            return self.engines[name]

    def status(self):
        return {n: {"loaded": n in self.engines,
                    "default": n == self.default,
                    "self_check": self.engines[n].self_check_result if n in self.engines else None,
                    "temperatures": (self.engines[n].calibration.temperatures
                                     if n in self.engines else None),
                    "max_options": self.models[n].get("max_options") or 16,
                    "reject_slot": bool(self.models[n].get("reject_slot", False))}
                for n in self.models}
