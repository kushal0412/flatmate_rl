"""FastAPI app for the Flatmate RL environment."""

from __future__ import annotations

import logging
import os

try:
    from openenv.core.env_server.web_interface import create_web_interface_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required. Install dependencies with `uv sync`.") from e

try:
    from flatmate_rl.models import FlatmateRlAction, FlatmateRlObservation
    from flatmate_rl.server.gradio_ui import build_flatmate_gradio_app
    from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment
except Exception:
    from models import FlatmateRlAction, FlatmateRlObservation
    from server.gradio_ui import build_flatmate_gradio_app
    from server.flatmate_rl_environment import FlatmateRlEnvironment


app = create_web_interface_app(
    FlatmateRlEnvironment,
    FlatmateRlAction,
    FlatmateRlObservation,
    env_name="flatmate_rl",
    gradio_builder=build_flatmate_gradio_app,
    max_concurrent_envs=4,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("flatmate_rl").setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    _configure_logging()
    ws_ping_interval = float(os.getenv("UVICORN_WS_PING_INTERVAL", "600"))
    ws_ping_timeout = float(os.getenv("UVICORN_WS_PING_TIMEOUT", "600"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        ws_ping_interval=ws_ping_interval,
        ws_ping_timeout=ws_ping_timeout,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
