#!/usr/bin/env python
"""Start the FireWatch API and dashboard.

Examples:
    python scripts/serve.py --config configs/app.yaml
    python scripts/serve.py --config configs/app.yaml --port 8080 --device 0

Open http://HOST:PORT/ for the dashboard. Stop with Ctrl-C or SIGTERM: running jobs are stopped
and recorded as interrupted with the results obtained so far.

Exit codes: 0 = clean shutdown, 2 = invalid configuration or the port is unavailable.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from firewatch.utils.config import ConfigError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config", help="app config YAML (default: $FIREWATCH_APP_CONFIG or built-in defaults)"
    )
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--device", help="'cpu' or a GPU index; overrides model.device")
    p.add_argument("--weights", help="checkpoint path; overrides model.weights")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from firewatch.app.config import load_app_config
    from firewatch.app.server import build_server

    overrides: dict = {}
    if a.host:
        overrides.setdefault("api", {})["host"] = a.host
    if a.port is not None:
        overrides.setdefault("api", {})["port"] = a.port
    if a.device:
        overrides.setdefault("model", {})["device"] = a.device
    if a.weights:
        overrides.setdefault("model", {})["weights"] = a.weights
    try:
        cfg = load_app_config(a.config, overrides)
        server, app = build_server(cfg)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    logging.getLogger("firewatch.app").info(
        "serving on http://%s:%s/  (model %s, data %s)", host, port, cfg.weights_path, cfg.data_dir
    )
    if app.detector.status()["state"] == "missing":
        logging.getLogger("firewatch.app").error(
            "checkpoint not found: %s - analysis requests will be refused (HTTP 503)",
            cfg.weights_path,
        )

    stopping = threading.Event()

    def _stop(signum, frame):
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        server.serve_forever()
    finally:
        server.app.manager.shutdown()
        server.app.store.close()
        server.server_close()
        logging.getLogger("firewatch.app").info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
