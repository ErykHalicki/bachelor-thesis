import threading

import numpy as np
from omegaconf import OmegaConf

from thesis.experiments.eval.chunking import RemoteDriver
from thesis.utils.wire import listen, recv_msg, send_msg

OBS_FEATURES = {"observation.state": {"dtype": "float32", "shape": (3,),
                                      "names": ["a", "b", "c"]}}
ACK = {"ok": True, "step": 7, "columns": ["observation.state"], "offsets": [-1, 0],
       "execute_len": 2, "action_offsets": [-3, -2, -1]}


def _server(sock, requests, chunks):
    """Stand-in for scripts/serve.py: record what arrives, reply with canned chunks."""
    conn, _ = sock.accept()
    with conn:
        for reply in [ACK] + [{"actions": c} for c in chunks]:
            requests.append(recv_msg(conn))
            send_msg(conn, reply)


def test_the_client_sends_the_action_history_it_actually_executed():
    sock = listen("127.0.0.1", 0)
    port = sock.getsockname()[1]
    requests = []
    chunks = [np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([[3.0, 3.0], [4.0, 4.0]])]
    thread = threading.Thread(target=_server, args=(sock, requests, chunks), daemon=True)
    thread.start()

    driver = RemoteDriver(
        OmegaConf.create({"server": f"127.0.0.1:{port}", "server_timeout": 10,
                          "image_size": None, "columns": {}, "slices": {},
                          "action_filter": {}, "execute_len": None}),
        OBS_FEATURES,
        run="e/p/abc",
    )
    assert driver.action_offsets == [-3, -2, -1]
    assert driver.action_len == 3
    assert driver.checkpoint_step == 7

    emitted = [driver.step({"observation.state": np.full(3, t, np.float32)})
               for t in range(4)]
    driver.close()
    thread.join(timeout=5)

    predicts = [r for r in requests if r.get("type") == "predict"]
    assert len(predicts) == 2
    assert predicts[0]["actions"] == [None, None, None]
    sent = predicts[1]["actions"]
    assert sent[0] is None
    assert [list(a) for a in sent[1:]] == [list(emitted[0]), list(emitted[1])]


def test_a_server_that_declares_no_action_offsets_buffers_nothing():
    sock = listen("127.0.0.1", 0)
    port = sock.getsockname()[1]
    requests = []
    ack = {k: v for k, v in ACK.items() if k != "action_offsets"}
    thread = threading.Thread(
        target=lambda: _server_with(sock, requests, ack), daemon=True)
    thread.start()

    driver = RemoteDriver(
        OmegaConf.create({"server": f"127.0.0.1:{port}", "server_timeout": 10,
                          "image_size": None, "columns": {}, "slices": {},
                          "action_filter": {}, "execute_len": None}),
        OBS_FEATURES,
        run="e/p/abc",
    )
    assert driver.action_offsets == [] and driver.action_len == 0
    driver.step({"observation.state": np.zeros(3, np.float32)})
    driver.close()
    thread.join(timeout=5)
    assert [r for r in requests if r.get("type") == "predict"][0]["actions"] == []


def _server_with(sock, requests, ack):
    conn, _ = sock.accept()
    with conn:
        for reply in [ack, {"actions": np.array([[1.0, 1.0], [2.0, 2.0]])}]:
            requests.append(recv_msg(conn))
            send_msg(conn, reply)
