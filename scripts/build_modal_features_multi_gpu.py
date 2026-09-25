"""Extract NEW modal features on multiple GPUs with one command.

Each GPU runs an independent shard in a child process. Once every shard
finishes, the metadata merger verifies complete coverage of the processed WAVs.
Run from the repository root with ``python -m scripts.build_modal_features_multi_gpu``.
"""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import monotonic, sleep


def _selected_devices(value):
    """Map logical GPU indices through an existing CUDA visibility mask."""
    try:
        indices = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--gpus must be comma-separated GPU indices") from error
    if (
        not indices
        or any(index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("--gpus needs distinct, nonnegative GPU indices")

    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return [(index, str(index)) for index in indices]
    visible = [part.strip() for part in mask.split(",") if part.strip()]
    if any(index >= len(visible) for index in indices):
        raise ValueError(
            f"Requested GPUs {indices}, but CUDA_VISIBLE_DEVICES exposes only "
            f"{len(visible)} device(s): {mask!r}"
        )
    return [(index, visible[index]) for index in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1", help="Visible GPU indices, e.g. 0,1")
    parser.add_argument(
        "--processed_root", type=Path, default=Path("../datasets/processed")
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--num_modes", type=int, default=128)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--gpu_cqt_batch_size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry_run", action="store_true", help="Print worker commands without running"
    )
    args = parser.parse_args()
    if args.num_modes < 1 or args.checkpoint_every < 1 or args.gpu_cqt_batch_size < 1:
        parser.error(
            "Mode count, checkpoint interval, and CQT batch size must be positive"
        )
    try:
        devices = _selected_devices(args.gpus)
    except ValueError as error:
        parser.error(str(error))

    repository = Path(__file__).resolve().parents[1]
    processed_root = (repository / args.processed_root).resolve()
    requested_output = args.out_dir or Path(
        f"../datasets/modal_features/processed_modal_new{args.num_modes}"
    )
    out_dir = (repository / requested_output).resolve()
    if not args.dry_run and not processed_root.is_dir():
        parser.error(f"Processed WAV directory does not exist: {processed_root}")

    worker_commands = []
    for shard_index, (logical_gpu, physical_gpu) in enumerate(devices):
        command = [
            sys.executable,
            "-m",
            "scripts.build_modal_features_new",
            "--processed_root",
            str(processed_root),
            "--out_dir",
            str(out_dir),
            "--num_modes",
            str(args.num_modes),
            "--compute_device",
            "cuda",
            "--num_shards",
            str(len(devices)),
            "--shard_index",
            str(shard_index),
            "--checkpoint_every",
            str(args.checkpoint_every),
            "--gpu_cqt_batch_size",
            str(args.gpu_cqt_batch_size),
        ]
        if args.resume:
            command.append("--resume")
        worker_commands.append((logical_gpu, physical_gpu, command))

    merge_command = [
        sys.executable,
        "-m",
        "scripts.merge_modal_shards",
        "--out_dir",
        str(out_dir),
        "--num_shards",
        str(len(devices)),
    ]
    if args.dry_run:
        for logical_gpu, physical_gpu, command in worker_commands:
            print(
                f"GPU {logical_gpu}: CUDA_VISIBLE_DEVICES={physical_gpu} {shlex.join(command)}"
            )
        if len(devices) > 1:
            print(f"Merge: {shlex.join(merge_command)}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    with ExitStack() as stack:
        try:
            for shard_index, (logical_gpu, physical_gpu, command) in enumerate(
                worker_commands
            ):
                log_path = out_dir / f"shard_{shard_index}.log"
                log = stack.enter_context(log_path.open("a", encoding="utf-8"))
                log.write(f"\nStarting GPU {logical_gpu}, shard {shard_index}\n")
                log.flush()
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = physical_gpu
                environment.setdefault("OMP_NUM_THREADS", "1")
                environment.setdefault("MKL_NUM_THREADS", "1")
                process = subprocess.Popen(
                    command,
                    cwd=repository,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                workers.append((shard_index, process, log_path))
                print(
                    f"Started GPU {logical_gpu} as shard {shard_index} "
                    f"(PID {process.pid}); log: {log_path}",
                    flush=True,
                )

            reported = set()
            next_status = monotonic() + 30
            while len(reported) < len(workers):
                for shard_index, process, log_path in workers:
                    status = process.poll()
                    if status is None or shard_index in reported:
                        continue
                    if status != 0:
                        raise RuntimeError(
                            f"Shard {shard_index} failed with exit code {status}; "
                            f"see {log_path}. Rerun with --resume after fixing it."
                        )
                    reported.add(shard_index)
                    print(f"Shard {shard_index} completed.", flush=True)
                if len(reported) == len(workers):
                    break
                if monotonic() >= next_status:
                    print(
                        f"Waiting for {len(workers) - len(reported)} shard(s)...",
                        flush=True,
                    )
                    next_status = monotonic() + 30
                sleep(2)
        except BaseException:
            for _, process, _ in workers:
                if process.poll() is None:
                    process.terminate()
            for _, process, _ in workers:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

    if len(devices) > 1:
        subprocess.run(merge_command, cwd=repository, check=True)
    print(f"Modal extraction complete: {out_dir}")


if __name__ == "__main__":
    main()
