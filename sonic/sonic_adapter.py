from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq


HEADER_SIZE = 1280
MAGIC = b"POSE"
PROTOCOL_VERSION = 1
FIELD_MAP = {
    "joint_pos": 1,
    "joint_vel": 2,
    "body_quat": 3,
    "frame_index": 4,
}
REQUIRED_FIELDS = ("joint_pos", "joint_vel", "body_quat", "frame_index")


def load_sonic_v1_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"SONIC bundle not found: {path}")
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def validate_sonic_v1_bundle(bundle: dict[str, np.ndarray]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIELDS if key not in bundle]
    if missing:
        raise ValueError(f"Missing required SONIC v1 fields: {missing}")

    joint_pos = np.asarray(bundle["joint_pos"])
    joint_vel = np.asarray(bundle["joint_vel"])
    body_quat = np.asarray(bundle["body_quat"])
    frame_index = np.asarray(bundle["frame_index"])

    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
        raise ValueError(f"joint_pos must have shape [N, 29], got {joint_pos.shape}")
    if joint_vel.ndim != 2 or joint_vel.shape[1] != 29:
        raise ValueError(f"joint_vel must have shape [N, 29], got {joint_vel.shape}")
    if body_quat.ndim != 2 or body_quat.shape[1] != 4:
        raise ValueError(f"body_quat must have shape [N, 4], got {body_quat.shape}")
    if frame_index.ndim != 1:
        raise ValueError(f"frame_index must have shape [N], got {frame_index.shape}")

    frame_count = joint_pos.shape[0]
    if joint_vel.shape[0] != frame_count or body_quat.shape[0] != frame_count or frame_index.shape[0] != frame_count:
        raise ValueError(
            "joint_pos, joint_vel, body_quat, and frame_index must share the same frame count"
        )

    for name, array in {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_quat": body_quat,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or inf values")

    return {
        "frame_count": int(frame_count),
        "joint_pos_shape": list(joint_pos.shape),
        "joint_vel_shape": list(joint_vel.shape),
        "body_quat_shape": list(body_quat.shape),
        "joint_pos_min": float(np.min(joint_pos)),
        "joint_pos_max": float(np.max(joint_pos)),
        "joint_vel_min": float(np.min(joint_vel)),
        "joint_vel_max": float(np.max(joint_vel)),
    }


def build_protocol_v1_frame(bundle: dict[str, np.ndarray], frame_id: int) -> dict[str, np.ndarray]:
    return {
        "joint_pos": np.asarray(bundle["joint_pos"][frame_id : frame_id + 1], dtype=np.float32),
        "joint_vel": np.asarray(bundle["joint_vel"][frame_id : frame_id + 1], dtype=np.float32),
        "body_quat": np.asarray(bundle["body_quat"][frame_id : frame_id + 1], dtype=np.float32),
        "frame_index": np.asarray(bundle["frame_index"][frame_id : frame_id + 1], dtype=np.int64),
    }


def pack_protocol_v1_message(frame_payload: dict[str, np.ndarray]) -> bytes:
    header = bytearray(HEADER_SIZE)
    struct.pack_into("4sHHI", header, 0, MAGIC, PROTOCOL_VERSION, 0, 1)

    descriptor_offset = 16
    struct.pack_into("I", header, 12, len(frame_payload))

    payload_offset = HEADER_SIZE
    payload_chunks: list[bytes] = []
    for field_name, array_value in frame_payload.items():
        array = np.asarray(array_value)
        if field_name == "frame_index":
            array = array.astype(np.int64, copy=False)
        elif array.dtype == np.float64:
            array = array.astype(np.float32)

        field_bytes = array.tobytes(order="C")
        payload_chunks.append(field_bytes)
        descriptor = (
            FIELD_MAP[field_name],
            array.ndim,
            *list(array.shape) + [1] * (4 - array.ndim),
            payload_offset,
            len(field_bytes),
        )
        struct.pack_into("II4III", header, descriptor_offset, *descriptor)
        descriptor_offset += 32
        payload_offset += len(field_bytes)

    return bytes(header) + b"".join(payload_chunks)


def publish_sonic_v1(
    *,
    bundle: dict[str, np.ndarray],
    host: str,
    port: int,
    topic: str,
    fps_override: float | None = None,
    speed: float = 1.0,
    dry_run: bool = False,
    loop: bool = False,
    log_every: int = 25,
) -> None:
    stats = validate_sonic_v1_bundle(bundle)
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")

    stream_fps = float(fps_override or np.asarray(bundle.get("fps", [50.0])).reshape(-1)[0])
    if stream_fps <= 0:
        raise ValueError(f"Invalid stream fps: {stream_fps}")
    frame_interval = 1.0 / (stream_fps * speed)
    frame_count = stats["frame_count"]

    if dry_run:
        for frame_id in range(frame_count):
            frame = build_protocol_v1_frame(bundle, frame_id)
            print(
                f"dry-run frame={frame_id} joint_pos_shape={list(frame['joint_pos'].shape)} "
                f"joint_vel_shape={list(frame['joint_vel'].shape)} body_quat={frame['body_quat'][0].tolist()}"
            )
        return

    endpoint = f"tcp://{host}:{port}"
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.bind(endpoint)
    topic_bytes = topic.encode("utf-8")

    start_time = time.perf_counter()
    sent_frames = 0
    print(f"Publishing SONIC protocol v1 on {endpoint} topic={topic} fps={stream_fps:.3f} speed={speed:.3f}")
    time.sleep(1.0)
    try:
        while True:
            for frame_id in range(frame_count):
                tick = time.perf_counter()
                frame = build_protocol_v1_frame(bundle, frame_id)
                socket.send(topic_bytes + b" " + pack_protocol_v1_message(frame))
                sent_frames += 1

                total_elapsed = max(time.perf_counter() - start_time, 1e-9)
                if frame_id == 0 or (frame_id + 1) % log_every == 0 or frame_id == frame_count - 1:
                    publish_rate = sent_frames / total_elapsed
                    print(
                        f"frame_id={frame_id} elapsed_sec={total_elapsed:.3f} "
                        f"publish_rate_fps={publish_rate:.2f}"
                    )

                spent = time.perf_counter() - tick
                if spent < frame_interval:
                    time.sleep(frame_interval - spent)

            if not loop:
                break
    finally:
        socket.close(0)
