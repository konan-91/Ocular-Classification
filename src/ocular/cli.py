"""Command line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import utils

DEFAULT_RAW = Path("data/raw")
DEFAULT_TOPOPLOTS = Path("data/topoplots")
DEFAULT_MANIFEST = Path("data/manifest.csv")
DEFAULT_ARTIFACTS = Path("artifacts")


def _add_prepare(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare", help="render the raw recordings as topoplots and write a manifest"
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_TOPOPLOTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--limit-per-event",
        type=int,
        default=None,
        help="cap segments per event type per recording, for a quick smoke run",
    )


def _add_train(subparsers) -> None:
    parser = subparsers.add_parser("train", help="train the classifier")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--architecture", default="resnet34")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="auto")


def _add_evaluate(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="score a trained model on the held out test split"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACTS / "model.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)


def _add_benchmark(subparsers) -> None:
    parser = subparsers.add_parser(
        "benchmark", help="compare the model against the ICA baseline"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACTS / "model.pt")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocular",
        description="Detect ocular artifacts in EEG from topographic scalp maps",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare(subparsers)
    _add_train(subparsers)
    _add_evaluate(subparsers)
    _add_benchmark(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    utils.configure_logging(args.verbose)

    if args.command == "prepare":
        from .prepare import prepare
        from . import manifest

        frame = prepare(args.raw, args.out, args.manifest, args.limit_per_event)
        print(manifest.summarise(frame))

    elif args.command == "train":
        from .train import TrainConfig, train

        train(
            args.manifest,
            args.out,
            TrainConfig(
                architecture=args.architecture,
                head_epochs=args.head_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                device=args.device,
            ),
        )

    elif args.command == "evaluate":
        from . import manifest, metrics, model as model_module
        from .benchmark import _split_from_metadata
        from .evaluate import evaluate

        frame = manifest.read(args.manifest)
        model, metadata = model_module.load(args.model)
        device = model_module.pick_device()

        frame = _split_from_metadata(frame, metadata)
        test_frame = frame[frame["split"] == "test"]

        scores = evaluate(
            model, test_frame, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        utils.write_json(scores, Path(args.out) / "test_metrics.json")
        metrics.plot_confusion_matrix(
            scores, Path(args.out) / "confusion_matrix_test.png",
            "Model, held out participants",
        )
        print(metrics.format_report("test", scores))

    elif args.command == "benchmark":
        from .benchmark import format_summary, run

        results = run(
            args.manifest, args.model, args.raw, args.out,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        print()
        print(format_summary(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
