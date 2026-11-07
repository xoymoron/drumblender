"""Serve fixed audition cards plus on-demand, unfiltered validation draws."""

import argparse
import json
import threading
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch

from drumblender.intervention import BETA, InterventionSAE, edit_direction
from scripts.audition_intervention import as_wave, save_waves, setup_folder
from scripts.control_intervention import RunData
from scripts.control_joint import sha256, write_json

GAINS = (0, -4, 4, -16, 16)
CLIENT = Path(__file__).with_name("control_audition_live.js")


def draw_index(count, excluded, seen, rng):
    pool = np.array([i for i in range(count) if i not in excluded], dtype=int)
    if not len(pool):
        raise ValueError("No validation samples remain outside the fixed examples.")
    available = pool[~np.isin(pool, list(seen))]
    restarted = not len(available)
    if restarted:
        seen.clear()
        available = pool
    index = int(rng.choice(available))
    seen.add(index)
    return index, restarted


class Engine:
    def __init__(self, args):
        self.args = args
        self.root = args.directory.resolve()
        self.run = RunData(args)
        self.sae_hash = sha256(args.run / "final.pt")
        payload = torch.load(
            args.run / "final.pt", map_location=args.device, weights_only=True
        )
        if payload["identity"] != self.run.identity or payload[
            "cache_sha256"
        ] != sha256(args.cache / "cache.json"):
            raise ValueError("Frozen checkpoint/cache/SAE identity mismatch.")
        fixed = json.loads((self.root / "protocol.json").read_text())
        if (
            fixed["identity"] != self.run.identity
            or fixed["sae_sha256"] != self.sae_hash
            or fixed["strike_code_updated"]
        ):
            raise ValueError(
                "Expected the matching fixed-strike audition for this SAE."
            )
        screening = json.loads((args.run / "screening.json").read_text())
        self.allowed = {
            "main": list(fixed["atoms"]),
            "comparison": list(fixed["baseline_atoms"]),
        }
        self.excluded = {
            int(r["atom"]): set(r["examples"]) for r in screening["candidates"]
        }
        self.floors = payload["floors"]
        self.nets = {}
        for name, key in (
            ("main", "state_dict"),
            ("comparison", "baseline_state_dict"),
        ):
            net = InterventionSAE().to(args.device).eval().requires_grad_(False)
            net.load_state_dict(payload[key])
            self.nets[name] = net
        self.rng = np.random.default_rng(self.run.seed + 2701)
        self.seen, self.last = {}, {}
        self.lock = threading.Lock()
        self.log = self.root / "live" / "draws.jsonl"
        self.log.parent.mkdir(exist_ok=True)
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                row = json.loads(line)
                key = (row["model"], row["atom"])
                seen = self.seen.setdefault(key, set())
                if row.get("cycle_restarted"):
                    seen.clear()
                seen.add(row["index"])
                if row["status"] == "complete":
                    self.last[key] = row
        write_json(
            self.root / "live" / "protocol.json",
            dict(
                identity=self.run.identity,
                sae_sha256=self.sae_hash,
                cache_sha256=payload["cache_sha256"],
                gains=GAINS,
                policy="Uniform validation draws without replacement per model/atom; exclude fixed examples only; no activation or sound-class filter; restart after exhaustion",
                seed=self.run.seed + 2701,
                strike_code="neutral and fixed",
                training=False,
                test_access=False,
            ),
        )

    def state(self):
        return dict(
            sae_sha256=self.sae_hash,
            allowed=self.allowed,
            draws=list(self.last.values()),
            validation_count=len(self.run.caches["val"].rows),
        )

    @torch.inference_mode()
    def render(self, model, atom):
        start = time.monotonic()
        cache = self.run.caches["val"]
        key = (model, atom)
        index, restarted = draw_index(
            len(cache.rows),
            self.excluded[atom],
            self.seen.setdefault(key, set()),
            self.rng,
        )
        row = dict(
            model=model,
            atom=atom,
            index=index,
            sample_id=cache.rows[index]["id"],
            pack=cache.rows[index]["pack"],
            cycle_restarted=restarted,
            utc=time.time(),
        )
        try:
            ref = self.run.reference("val", index)
            net = self.nets[model]
            codes = self.run.codes(net, ref)
            direction, radius = edit_direction(net, codes, atom, self.floors[atom])
            neutral = self.run.model.decode(ref, checkpoint_activations=False)
            scopes = {
                k: {} for k in ("mix", "modal", "noise", "transient", "modal-only")
            }
            for alpha in GAINS:
                edited = ref.h_ref + self.run.std * BETA * alpha * direction
                result = (
                    neutral
                    if alpha == 0
                    else self.run.model.decode(
                        ref, edited, checkpoint_activations=False
                    )
                )
                label = "0" if alpha == 0 else f"{alpha:+d}x"
                for scope, wave in dict(
                    mix=result.waveform,
                    modal=result.modal_waveform,
                    noise=result.noise_waveform,
                    transient=result.transient_waveform,
                    **{
                        "modal-only": result.modal_waveform
                        + neutral.noise_waveform
                        + neutral.transient_waveform
                    },
                ).items():
                    scopes[scope][label] = as_wave(wave, ref)
            job = uuid.uuid4().hex
            folder = self.root / "live" / job
            setup_folder(folder)
            links = {
                scope: {
                    label: f"live/{job}/{path}"
                    for label, path in save_waves(folder, scope, waves).items()
                }
                for scope, waves in scopes.items()
            }
            row.update(
                status="complete",
                active=bool(codes[..., atom].any()),
                radius=float(radius),
                floor_hit=bool(radius < self.floors[atom]),
                seconds=time.monotonic() - start,
                audio=links,
                name=self.run.datasets["val"]
                .metadata[row["sample_id"]]
                .get("orig_relpath", row["sample_id"]),
            )
            self.last[key] = row
        except Exception as exc:
            row.update(status="error", error=str(exc), seconds=time.monotonic() - start)
            raise
        finally:
            with self.log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
        return row


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, engine, **kwargs):
        self.engine = engine
        super().__init__(*args, directory=str(engine.root), **kwargs)

    def reply(self, code, payload, content_type="application/json"):
        body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/state":
            return self.reply(200, self.engine.state())
        if path == "/control-live.js":
            return self.reply(
                200, CLIENT.read_bytes(), "text/javascript; charset=utf-8"
            )
        if path in (
            "/",
            "/index.html",
            "/modal.html",
            "/noise.html",
            "/modal-only.html",
            "/baseline.html",
            "/baseline-modal.html",
            "/baseline-noise.html",
        ):
            page = self.engine.root / ("index.html" if path == "/" else path[1:])
            if page.is_file():
                text = page.read_text().replace(
                    "</html>", '<script src="/control-live.js"></script></html>'
                )
                return self.reply(200, text.encode(), "text/html; charset=utf-8")
        return super().do_GET()

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self.reply(403, dict(error="Same-origin requests only."))
        if urlsplit(self.path).path != "/api/render":
            return self.reply(404, dict(error="Unknown endpoint."))
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size.")
            request = json.loads(self.rfile.read(size))
            model, atom = request["model"], request["atom"]
            if type(atom) is not int or atom not in self.engine.allowed.get(model, []):
                raise ValueError("Choose an atom on this audition page.")
        except (ValueError, KeyError, TypeError) as exc:
            return self.reply(400, dict(error=str(exc)))
        if not self.engine.lock.acquire(blocking=False):
            return self.reply(
                409,
                dict(error="Another render is running. Try again when it finishes."),
            )
        try:
            result = self.engine.render(model, atom)
        except Exception as exc:
            print(f"Render failed: {exc}", flush=True)
            return self.reply(500, dict(error=str(exc)))
        finally:
            self.engine.lock.release()
        return self.reply(200, result)

    def list_directory(self, path):
        self.send_error(403, "Directory listing is disabled.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "noise-bank", "data-dir", "cache", "run", "directory"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--meta-file", default="metadata.json")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    engine = Engine(args)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(Handler, engine=engine)
    )
    print(
        f"READY http://127.0.0.1:{args.port}/index.html | validation only | {args.device}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
