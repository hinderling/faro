import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import useq

from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    try:
        from PIL import Image

        HAS_PIL = True
    except ImportError:
        HAS_PIL = False


class UC2(AbstractMicroscope):
    """Microscope controller that forwards MDA sequences to an ImSwitch/UC2 server.

    The server is expected to expose the ExperimentController REST API used in
    ``uc2_mda_demo.py`` (``get_mda_capabilities``, ``get_mda_sequence_info``,
    ``start_mda_experiment``). This class converts a ``df_acquire`` schedule
    into a :class:`useq.MDASequence` and posts it to the running server.
    """

    CHANNEL_GROUP = "Channel"
    USE_ONLY_PFS = False  # UC2/ImSwitch does not require PFS-only z handling

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        timeout: float = 10.0,
        webhook_port: int = 9000,
    ):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.api_base = f"{self.server_url}/ExperimentController"
        self.timeout = timeout
        self.capabilities: Dict[str, Any] = {}
        self.webhook_port = webhook_port
        self.webhook_url = f"http://localhost:{webhook_port}/frames"
        self.webhook_server: Optional[HTTPServer] = None
        self.webhook_thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self._current_df_acquire: Optional[pd.DataFrame] = None
        self.init_scope()

    def init_scope(self):
        """Check that the UC2/ImSwitch server is reachable and cache capabilities."""
        try:
            self.capabilities = self.check_capabilities()
            if not self.capabilities:
                print("UC2: capabilities request returned empty response")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"UC2: failed to reach server at {self.api_base}: {exc}")

    def check_capabilities(self) -> Dict[str, Any]:
        """Call ``get_mda_capabilities`` on the server."""
        response = requests.get(
            f"{self.api_base}/get_mda_capabilities", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def configure_webhook(self) -> bool:
        """Configure ImSwitch to send frame notifications to our webhook."""
        try:
            response = requests.post(
                f"{self.api_base}/setMDAWebhookUrl", params={"url": self.webhook_url}
            )
            response.raise_for_status()
            config = response.json()

            if config.get("success"):
                print(f"UC2: Webhook configured: {config.get('webhook_url')}")

                # Test webhook connectivity
                test_response = requests.post(f"{self.api_base}/testMDAWebhook")
                test_response.raise_for_status()
                test_result = test_response.json()

                if test_result.get("success"):
                    print(
                        f"UC2: Webhook test passed (status: {test_result.get('status_code')})"
                    )
                    return True
                else:
                    print(f"UC2: Webhook test failed: {test_result.get('error')}")
                    return False
            else:
                print("UC2: Failed to configure webhook")
                return False

        except Exception as e:
            print(f"UC2: Error configuring webhook: {e}")
            return False

    def get_webhook_config(self) -> Dict[str, Any]:
        """Get current webhook configuration from ImSwitch."""
        try:
            response = requests.get(f"{self.api_base}/getMDAWebhookConfig")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"UC2: Error getting webhook config: {e}")
            return {}

    def start_webhook_server(self) -> bool:
        """Start local HTTP server to receive frame notifications."""
        if self.webhook_server is not None:
            return True

        print(f"UC2: Starting webhook server on port {self.webhook_port}...")

        # Create handler class with reference to this UC2 instance
        uc2_instance = self

        class WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/frames":
                    content_length = int(self.headers["Content-Length"])
                    body = self.rfile.read(content_length)

                    try:
                        data = json.loads(body)
                        uc2_instance._handle_frame_notification(data)

                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "received"}).encode())
                    except Exception as e:
                        print(f"UC2: Error processing webhook: {e}")
                        self.send_response(500)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                """Suppress default HTTP server logging."""
                pass

        try:
            self.webhook_server = HTTPServer(("localhost", self.webhook_port), WebhookHandler)
            self.webhook_thread = threading.Thread(
                target=self.webhook_server.serve_forever, daemon=True
            )
            self.webhook_thread.start()
            print(f"UC2: Webhook server running at {self.webhook_url}")
            return True
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"UC2: Port {self.webhook_port} is already in use!")
                print(
                    f"UC2: Try a different port or stop the process using: lsof -ti:{self.webhook_port} | xargs kill"
                )
            else:
                print(f"UC2: Failed to start webhook server: {e}")
            self.webhook_server = None
            return False
        except Exception as e:
            print(f"UC2: Unexpected error starting webhook server: {e}")
            self.webhook_server = None
            return False

    def stop_webhook_server(self):
        """Stop the webhook server."""
        if self.webhook_server:
            self.webhook_server.shutdown()
            self.webhook_server = None
            print("UC2: Webhook server stopped")

    def _handle_frame_notification(self, data: Dict[str, Any]):
        """Handle incoming frame notification from ImSwitch."""
        if data.get("test"):
            print(f"UC2: Webhook test received: {data.get('message')}")
            return

        filepath = data.get("filepath", "")
        if not filepath or not Path(filepath).exists():
            print(f"UC2: Frame file not found: {filepath}")
            return

        self.frame_count += 1
        event_index = data.get("event_index", {})
        channel = data.get("channel", "")

        print(
            f"UC2: Frame {self.frame_count}: t={event_index.get('t')}, p={event_index.get('p')}, "
            f"c={event_index.get('c')}, z={event_index.get('z')}, channel={channel}"
        )

        # Load the image and call pipeline if available
        if self.pipeline is not None:
            try:
                # Load image
                if HAS_TIFFFILE:
                    img = tifffile.imread(filepath)
                elif HAS_PIL:
                    img = np.array(Image.open(filepath))
                else:
                    print("UC2: No image loading library available (install tifffile)")
                    return

                # Ensure image has channel dimension (pipeline expects 3D: CYX)
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]  # Add channel dimension

                # Build metadata from notification
                metadata = {
                    "img_type": "raw",
                    "timestep": event_index.get("t", 0),
                    "fov": event_index.get("p", 0),
                    "channel_idx": event_index.get("c", 0),
                    "z": event_index.get("z", 0),
                    "channel": channel,
                    "fname": Path(filepath).stem,
                    "filepath": filepath,
                    "stim": False,  #TODO: implement SLM
                }

                # Add df_acquire metadata if available
                if self._current_df_acquire is not None:
                    try:
                        row_mask = (
                            (self._current_df_acquire["timestep"] == metadata["timestep"])
                            & (self._current_df_acquire["fov"] == metadata["fov"])
                        )
                        matching_rows = self._current_df_acquire[row_mask]
                        if len(matching_rows) > 0:
                            row = matching_rows.iloc[0]
                            metadata.update(
                                {
                                    "fov_object": row.get("fov_object"),
                                    "fov_x": row.get("fov_x"),
                                    "fov_y": row.get("fov_y"),
                                    "fov_z": row.get("fov_z"),
                                    "time": row.get("time"),
                                    "cell_line": row.get("cell_line"),
                                    "channels": row.get("channels"),
                                }
                            )
                    except Exception as e:
                        print(f"UC2: Could not enrich metadata from df_acquire: {e}")

                # Create a minimal MDAEvent-like object for the pipeline
                class FrameEvent:
                    def __init__(self, meta):
                        self.metadata = meta
                        self.index = {
                            "t": meta.get("timestep", 0),
                            "p": meta.get("fov", 0),
                            "c": meta.get("channel_idx", 0),
                            "z": meta.get("z", 0),
                        }

                event = FrameEvent(metadata)

                # Call the pipeline in a separate thread to avoid blocking webhook
                threading.Thread(
                    target=self.pipeline.run, args=(img, event), daemon=True
                ).start()

            except Exception as e:
                print(f"UC2: Error processing frame {filepath}: {e}")

    def preview_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the server to summarize an experiment (ImSwitch simplified schema)."""
        response = requests.post(
            f"{self.api_base}/get_mda_sequence_info",
            json=experiment,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _df_to_imswitch_experiment(self, df_acquire: pd.DataFrame) -> Dict[str, Any]:
        """Build simplified experiment payload for ImSwitch/UC2 server.

        Keys:
        - channels: list of {name, exposure, power}
        - time_points: number of unique timesteps
        - time_interval: median spacing of unique times (seconds)
        - positions: list of {x,y,z} from unique FOVs
        - experiment_name: label (from df if present)
        """
        if df_acquire is None or len(df_acquire) == 0:
            raise ValueError("df_acquire is empty; nothing to send to UC2 server")

        # Channels from first row
        first_channels = df_acquire.iloc[0].get("channels", [])
        channels: List[Dict[str, Any]] = []
        for ch in first_channels:
            channels.append(
                {
                    "name": ch.get("name"),
                    "exposure": ch.get("exposure"),
                    "power": ch.get("power"),
                }
            )

        # Time info
        unique_times = sorted({float(t) for t in df_acquire["time"].unique()})
        time_points = len(unique_times)
        if time_points > 1:
            diffs = [j - i for i, j in zip(unique_times[:-1], unique_times[1:])]
            diffs_sorted = sorted(diffs)
            mid = len(diffs_sorted) // 2
            if len(diffs_sorted) % 2 == 0:
                time_interval = (diffs_sorted[mid - 1] + diffs_sorted[mid]) / 2.0
            else:
                time_interval = diffs_sorted[mid]
        else:
            time_interval = 0.0

        # Positions: one per unique FOV
        positions: List[Dict[str, float]] = []
        seen = set()
        for _, row in df_acquire.iterrows():
            fov = int(row.get("fov", 0))
            if fov in seen:
                continue
            seen.add(fov)
            positions.append(
                {
                    "x": float(row.get("fov_x", 0) or 0),
                    "y": float(row.get("fov_y", 0) or 0),
                    "z": float(row.get("fov_z", 0) or 0),
                }
            )

        experiment: Dict[str, Any] = {
            "channels": channels,
            "time_points": time_points,
            "time_interval": time_interval,
            "positions": positions,
            "experiment_name": str(df_acquire.iloc[0].get("cell_line", "UC2_Run")),
        }

        return experiment

    def run_experiment(
        self,
        df_acquire: pd.DataFrame,
        *,
        wait_for_completion: bool = True,
        request_timeout: Optional[float] = None,
        do_preview: bool = False,
        save_directory: Optional[str] = None,
        enable_webhook: bool = True,
    ) -> Dict[str, Any]:
        """Send the acquisition schedule to the UC2/ImSwitch server and start it.

        When wait_for_completion=False, returns immediately after starting with a
        short timeout so the server can run the workflow asynchronously.

        If enable_webhook=True (default), starts a local webhook server to receive
        frame notifications and call the pipeline on each saved image.
        """
        # Store df_acquire for metadata enrichment in frame callbacks
        self._current_df_acquire = df_acquire

        # Start webhook server and configure ImSwitch if requested
        if enable_webhook and self.pipeline is not None:
            if not self.start_webhook_server():
                print("UC2: Continuing without webhook notifications...")
            else:
                if not self.configure_webhook():
                    print("UC2: Webhook configuration failed, continuing anyway...")

        experiment = self._df_to_imswitch_experiment(df_acquire)

        preview_info: Optional[Dict[str, Any]] = None
        if do_preview:
            try:
                preview_info = self.preview_experiment(experiment)
            except Exception as exc:
                print(
                    f"UC2: preview failed, continuing to start experiment anyway: {exc}"
                )

        # include optional server-side save directory if provided
        payload = dict(experiment)
        if save_directory:
            payload["save_directory"] = save_directory
        timeout = request_timeout if request_timeout is not None else (
            0.8 if not wait_for_completion else self.timeout
        )
        try:
            response = requests.post(
                f"{self.api_base}/start_mda_experiment",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ReadTimeout:
            if not wait_for_completion:
                result = {"started": True, "detail": "Workflow started (async)"}
            else:
                raise
        except requests.HTTPError as exc:
            body = None
            try:
                body = response.text
            except Exception:
                body = None
            msg = f"UC2 start_mda_experiment failed: {exc}"
            if body:
                msg += f"\nResponse body: {body}"
            raise requests.HTTPError(msg) from exc

        if preview_info is not None:
            result["preview"] = preview_info
        return result

    def post_experiment(self):
        """Clean up after experiment: stop webhook server and clear references."""
        self.stop_webhook_server()
        self._current_df_acquire = None
        print(f"UC2: Experiment complete. Total frames processed: {self.frame_count}")
