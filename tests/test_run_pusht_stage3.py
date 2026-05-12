import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_pusht_stage3.py"


def load_stage3_module(stablewm_home: Path):
    module_name = "test_run_pusht_stage3_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict("os.environ", {"STABLEWM_HOME": str(stablewm_home)}):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def write_eval_result(path: Path) -> None:
    metrics = {
        "episode_successes": [True, False],
        "eval_row_indices": [1, 2],
        "eval_episodes": [10, 11],
        "eval_start_idx": [20, 21],
        "eval_manifest": {"split": "val"},
        "normalizer_metadata": {"normalizer_scope": "train"},
        "solver_batch_size": 128,
    }
    path.write_text(
        "metrics_json: " + json.dumps(metrics) + "\n" + "evaluation_time: 1.0 seconds\n",
        encoding="utf-8",
    )


class RunPushtStage3Tests(unittest.TestCase):
    def test_write_report_uses_large_val_manifest_count(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="stage3-report-test-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module = load_stage3_module(tmp_path)
            variant = module.VARIANTS["baseline"]
            name = module.output_name(variant, 3072, False, "")
            run_dir = tmp_path / "runs" / "pusht_expert_train" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            write_eval_result(run_dir / module.eval_filename("val", 50, 500))
            (run_dir / f"{name}_epoch_50_object.ckpt").write_text(
                "ckpt",
                encoding="utf-8",
            )

            report_dir = tmp_path / "report"
            with mock.patch.object(module, "REPORT_DIR", report_dir), mock.patch.object(
                module,
                "RUNS_ROOT",
                tmp_path / "runs" / "pusht_expert_train",
            ):
                module.write_report(
                    [variant],
                    [3072],
                    [5, 50],
                    smoke=False,
                    val_manifest_kind="large",
                )

            rows = json.loads(
                (report_dir / "pusht_stage3_v1_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(rows[0]["best_val_epoch"], 50)
            self.assertTrue(rows[0]["final_ckpt_exists"])

    def test_main_train_mode_uses_terminal_epoch(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="stage3-main-test-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module = load_stage3_module(tmp_path)
            output = io.StringIO()

            def fake_run_command(cmd, *, dry_run):
                print(" ".join(cmd), file=output)
                return 0

            argv = [
                "run_pusht_stage3.py",
                "--mode",
                "train",
                "--ids",
                "baseline",
                "--train-seeds",
                "3072",
                "--epochs",
                "5",
                "50",
                "--dry-run",
            ]
            with mock.patch.object(module, "require_dataset"), mock.patch.object(
                module,
                "run_command",
                side_effect=fake_run_command,
            ), mock.patch.object(sys, "argv", argv):
                module.main()

            self.assertIn("trainer.max_epochs=50", output.getvalue())

    def test_selected_test_eval_uses_best_val_epoch(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="stage3-selected-test-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module = load_stage3_module(tmp_path)
            calls = []

            def fake_eval_variant(*args, **kwargs):
                calls.append(
                    {
                        "split": kwargs["split"],
                        "epoch": kwargs["epoch"],
                        "manifest": kwargs["manifest"],
                    }
                )
                return True

            with mock.patch.object(
                module,
                "_best_epoch_for",
                return_value=(50, {"success_percent": 50.0}),
            ), mock.patch.object(module, "eval_variant", side_effect=fake_eval_variant):
                ok = module.run_selected_test_evals(
                    [module.VARIANTS["baseline"]],
                    [3072],
                    [5, 50],
                    manifests={"test": Path("test-manifest.json")},
                    dry_run=False,
                    force=False,
                    smoke=False,
                    num_samples=None,
                    n_steps=None,
                    val_manifest_kind="large",
                )

            self.assertTrue(ok)
            self.assertEqual(
                calls,
                [{"split": "test", "epoch": 50, "manifest": Path("test-manifest.json")}],
            )


if __name__ == "__main__":
    unittest.main()
