"""trainer.evaluation — evaluación por época: predict -> formato común -> métrica nuestra."""

from trainer.evaluation.evaluate import (
    DEFAULT_THRESHOLDS,
    EvalShardDataset,
    benchmark_shard_uris,
    calibrate,
    categories_for,
    eval_collate,
    evaluate,
    load_categories,
    make_eval_loader,
    predict_dataset,
    run_eval,
    to_lines_file,
)

__all__ = ["DEFAULT_THRESHOLDS", "to_lines_file", "eval_collate", "EvalShardDataset",
           "make_eval_loader", "predict_dataset", "categories_for", "evaluate", "calibrate",
           "run_eval", "load_categories", "benchmark_shard_uris"]
