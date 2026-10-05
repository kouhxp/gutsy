"""gutsy-inference command line.

  gutsy-inference serve   --config models.json [--host 127.0.0.1] [--port 8765] [--cors]
  gutsy-inference decide  --config models.json request.json      (or - for stdin)
  gutsy-inference bench   --config models.json [--model gutsy-2b]
  gutsy-inference check   --config models.json                   (load + self-check all models)
"""
import argparse
import json
import os
import sys

from .registry import Registry


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gutsy-inference")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "decide", "bench", "check"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="models.json")
        p.add_argument("--threads", type=int, help="CPU threads (default: physical-core estimate)")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8765)
            p.add_argument("--api-key-env", help="require 'Authorization: Bearer $VAR'")
            p.add_argument("--no-preload", action="store_true")
            p.add_argument("--quiet", action="store_true")
            p.add_argument("--cors", action="store_true",
                           help="allow browser pages from any origin (needed by examples/)")
        if name == "decide":
            p.add_argument("request", help="request JSON file, or - for stdin")
        if name == "bench":
            p.add_argument("--model")
            p.add_argument("--state-tokens", type=int, nargs="+", default=[200, 1000, 4000])
            p.add_argument("--questions", type=int, default=5)
            p.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(argv)
    reg = Registry(a.config, n_threads=a.threads)

    if a.cmd == "serve":
        key = None
        if a.api_key_env:
            key = os.environ.get(a.api_key_env)
            if not key:
                sys.exit(f"environment variable {a.api_key_env} is not set")
        from .server import serve
        serve(reg, a.host, a.port, key, preload=not a.no_preload, quiet=a.quiet, cors=a.cors)
    elif a.cmd == "decide":
        body = json.load(sys.stdin if a.request == "-" else open(a.request))
        print(json.dumps(reg.get(body.get("model")).decide(body), indent=2, ensure_ascii=False))
    elif a.cmd == "bench":
        from .bench import run
        e = reg.get(a.model)
        print(f"threads: {getattr(e.backend, 'n_threads', '?')}")
        print(run(e, a.state_tokens, a.questions, a.repeats))
    elif a.cmd == "check":
        for n in reg.names():
            e = reg.get(n)
            print(f"{n}: {e.self_check_result}; temperatures {e.calibration.temperatures}")


if __name__ == "__main__":
    main()
