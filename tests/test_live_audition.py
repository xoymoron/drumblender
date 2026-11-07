import http.client
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np

from scripts.serve_control_audition import Handler, draw_index


def test_random_draws_cover_all_unselected_samples_without_activation_filter():
    rng, seen = np.random.default_rng(17), set()
    draws = [draw_index(7, {0, 2, 5}, seen, rng) for _ in range(4)]
    assert {i for i, _ in draws} == {1, 3, 4, 6}
    assert not any(restarted for _, restarted in draws)
    index, restarted = draw_index(7, {0, 2, 5}, seen, rng)
    assert restarted and index in {1, 3, 4, 6} and seen == {index}


def test_live_page_keeps_fixed_cards_and_api_validates_inputs(tmp_path):
    page = "<html><section>fixed example</section></html>"
    (tmp_path / "index.html").write_text(page)
    calls = []

    def render(model, atom):
        calls.append((model, atom))
        return dict(active=False, atom=atom, index=9)

    engine = SimpleNamespace(
        root=tmp_path,
        allowed={"main": [22]},
        lock=threading.Lock(),
        render=render,
        state=lambda: dict(draws=[]),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, engine=engine))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    conn = http.client.HTTPConnection(*server.server_address)

    def request(path, payload, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        conn.request("POST", path, json.dumps(payload), headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    try:
        conn.request("GET", "/index.html")
        response = conn.getresponse()
        text = response.read().decode()
        assert (
            response.status == 200
            and "fixed example" in text
            and "/control-live.js" in text
        )
        assert (tmp_path / "index.html").read_text() == page
        status, payload = request("/api/render", dict(model="main", atom=22))
        assert status == 200 and payload["active"] is False and calls == [("main", 22)]
        assert request("/api/render", dict(model="main", atom=69))[0] == 400
        assert request("/api/render", dict(model="main", atom=True))[0] == 400
        assert (
            request(
                "/api/render", dict(model="main", atom=22), "http://unrelated.example"
            )[0]
            == 403
        )
        engine.lock.acquire()
        try:
            assert request("/api/render", dict(model="main", atom=22))[0] == 409
        finally:
            engine.lock.release()
        assert calls == [("main", 22)]
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        worker.join()
