#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sonic_adapter import load_sonic_v1_bundle, publish_sonic_v1, validate_sonic_v1_bundle


COMMON_BUNDLE_RELATIVE_PATHS = (
    Path("sonic_stream_v1.npz"),
    Path("sonic_v1") / "sonic_stream_v1.npz",
)
LEGACY_BUNDLE_GLOBS = ("*_sonic_v1_input.npz",)


def resolve_bundle_path(input_path: Path) -> Path:
    resolved = input_path.expanduser()
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Input path not found: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Expected a file or directory, got {resolved}")

    for relative_path in COMMON_BUNDLE_RELATIVE_PATHS:
        candidate = resolved / relative_path
        if candidate.exists():
            return candidate

    matches: list[Path] = []
    matches.extend(sorted(resolved.rglob("sonic_stream_v1.npz")))
    for pattern in LEGACY_BUNDLE_GLOBS:
        matches.extend(sorted(resolved.rglob(pattern)))

    deduped_matches: list[Path] = []
    seen: set[Path] = set()
    for match in matches:
        normalized = match.resolve()
        if normalized not in seen:
            seen.add(normalized)
            deduped_matches.append(normalized)

    if len(deduped_matches) == 1:
        return deduped_matches[0]
    if not deduped_matches:
        raise FileNotFoundError(
            "Could not find a SONIC v1 bundle under the input directory. "
            "Expected sonic_stream_v1.npz or sonic_v1/sonic_stream_v1.npz."
        )
    raise RuntimeError(
        "Found multiple candidate SONIC v1 bundles. "
        f"Please pass the exact file path instead: {[str(path) for path in deduped_matches]}"
    )


def build_validation_payload(bundle_path: Path, stats: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "input": str(bundle_path.resolve()), **stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or publish a SONIC v1 bundle from a pipeline output directory or a direct .npz path."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to sonic_stream_v1.npz or to a run directory that contains it.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="ZMQ publisher bind host")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ publisher port")
    parser.add_argument("--topic", default="pose", help="ZMQ topic")
    parser.add_argument("--fps-override", type=float, default=None, help="Optional publish fps override")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop forever while publishing")
    parser.add_argument("--dry-run", action="store_true", help="Print frame summaries instead of sending ZMQ messages")
    parser.add_argument("--publish", action="store_true", help="Publish the stream after validation")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a saved validation verdict JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = resolve_bundle_path(args.input)
    bundle = load_sonic_v1_bundle(bundle_path)
    stats = validate_sonic_v1_bundle(bundle)
    verdict = build_validation_payload(bundle_path, stats)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Resolved SONIC bundle: {bundle_path.resolve()}")
    print(f"Frame count: {stats['frame_count']}")
    print(f"joint_pos shape: {stats['joint_pos_shape']}")
    print(f"joint_vel shape: {stats['joint_vel_shape']}")
    print(f"body_quat shape: {stats['body_quat_shape']}")
    if args.output_json is not None:
        print(f"Validation JSON: {args.output_json.resolve()}")

    if not args.publish and not args.dry_run:
        print("Validation passed. Use --dry-run or --publish to continue.")
        return

    publish_sonic_v1(
        bundle=bundle,
        host=args.host,
        port=args.port,
        topic=args.topic,
        fps_override=args.fps_override,
        speed=args.speed,
        dry_run=args.dry_run,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
