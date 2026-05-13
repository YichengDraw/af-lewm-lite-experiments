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
            curve_rows = json.loads(
                (report_dir / "pusht_stage3_v1_val_curve.json").read_text(encoding="utf-8")
            )
            self.assertEqual(rows[0]["best_val_epoch"], 50)
            self.assertTrue(rows[0]["final_ckpt_exists"])
            self.assertEqual(curve_rows[-1]["epoch"], 50)
            self.assertEqual(curve_rows[-1]["val_success_percent"], 50.0)

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

    def test_eval_mode_can_override_solver_batch_size(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="stage3-eval-batch-test-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module = load_stage3_module(tmp_path)
            output = io.StringIO()
            name = module.output_name(module.VARIANTS["baseline"], 3072, False, "")
            run_dir = tmp_path / "runs" / "pusht_expert_train" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / f"{name}_epoch_50_object.ckpt").write_text(
                "ckpt",
                encoding="utf-8",
            )
            manifest_dir = tmp_path / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest = manifest_dir / "test.json"
            manifest.write_text(json.dumps({"num_eval": 1000, "rows": []}), encoding="utf-8")

            def fake_run_command(cmd, *, dry_run):
                print(" ".join(cmd), file=output)
                return 0

            with mock.patch.object(module, "run_command", side_effect=fake_run_command):
                ok = module.eval_variant(
                    module.VARIANTS["baseline"],
                    3072,
                    split="test",
                    manifest=manifest,
                    epoch=50,
                    dry_run=False,
                    force=False,
                    smoke=False,
                    solver_batch_size=100,
                )

            self.assertTrue(ok)
            self.assertIn("solver.batch_size=100", output.getvalue())

    def test_large_test_eval_is_chunked_and_aggregated(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="stage3-chunk-test-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module = load_stage3_module(tmp_path)
            variant = module.VARIANTS["baseline"]
            name = module.output_name(variant, 3072, False, "")
            run_dir = tmp_path / "runs" / "pusht_expert_train" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / f"{name}_epoch_50_object.ckpt").write_text(
                "ckpt",
                encoding="utf-8",
            )
            rows = [
                {"row_index": 10 + idx, "episode_idx": 100 + idx, "start_step": 20 + idx}
                for idx in range(5)
            ]
            manifest_dir = tmp_path / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest = manifest_dir / "test_n5.json"
            manifest.write_text(
                json.dumps({"num_eval": 5, "split": "test", "seed": 9200, "rows": rows}),
                encoding="utf-8",
            )

            def fake_run_command(cmd, *, dry_run):
                manifest_arg = next(item for item in cmd if item.startswith("eval.manifest_path="))
                filename_arg = next(item for item in cmd if item.startswith("output.filename="))
                chunk_manifest = Path(manifest_arg.split("=", 1)[1])
                filename = filename_arg.split("=", 1)[1]
                chunk_rows = json.loads(chunk_manifest.read_text(encoding="utf-8"))["rows"]
                flags = [row["row_index"] % 2 == 0 for row in chunk_rows]
                metrics = {
                    "success_rate": 100.0 * sum(flags) / len(flags),
                    "episode_successes": flags,
                    "seeds": list(range(len(flags))),
                    "eval_row_indices": [row["row_index"] for row in chunk_rows],
                    "eval_episodes": [row["episode_idx"] for row in chunk_rows],
                    "eval_start_idx": [row["start_step"] for row in chunk_rows],
                    "normalizer_metadata": {"normalizer_scope": "train_episodes_from_split_metadata"},
                    "eval_manifest": {"split": "test", "size": len(chunk_rows)},
                    "solver_batch_size": 50,
                }
                (run_dir / filename).write_text(
                    "metrics_json: " + json.dumps(metrics) + "\n"
                    "evaluation_time: 2.5 seconds\n",
                    encoding="utf-8",
                )
                return 0

            with mock.patch.object(
                module,
                "MANIFEST_DIR",
                manifest_dir,
            ), mock.patch.object(module, "run_command", side_effect=fake_run_command):
                ok = module.eval_variant(
                    variant,
                    3072,
                    split="test",
                    manifest=manifest,
                    epoch=50,
                    dry_run=False,
                    force=False,
                    smoke=False,
                    test_chunk_size=2,
                )

            self.assertTrue(ok)
            final_result = module.parse_eval_result(run_dir / "stage3_test_epoch50_num5.txt")
            self.assertEqual(final_result["episodes"], 5)
            self.assertEqual(final_result["successes"], 3)
            self.assertEqual(final_result["success_percent"], 60.0)
            self.assertEqual(final_result["eval_row_indices"], [10, 11, 12, 13, 14])

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

    def test_run_cycle_dry_run_still_schedules_selected_test(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="stage3-cycle-dry-run-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module = load_stage3_module(tmp_path)

            with mock.patch.object(module, "train_variant", return_value=True) as train_mock, mock.patch.object(
                module,
                "eval_variant",
                return_value=True,
            ) as eval_mock, mock.patch.object(
                module,
                "run_selected_test_evals",
                return_value=True,
            ) as selected_test_mock, mock.patch.object(module, "write_report") as write_report_mock:
                ok = module.run_cycle(
                    [module.VARIANTS["baseline"]],
                    [3072],
                    [5, 50],
                    manifests={
                        "val_small": Path("val-small.json"),
                        "val_large": Path("val-large.json"),
                        "test": Path("test.json"),
                    },
                    dry_run=True,
                    force=False,
                    smoke=False,
                    wandb=False,
                    batch_size=None,
                    val_manifest_kind="small",
                    num_samples=None,
                    n_steps=None,
                )

            self.assertTrue(ok)
            self.assertEqual(train_mock.call_count, 2)
            self.assertEqual(eval_mock.call_count, 2)
            selected_test_mock.assert_called_once()
            self.assertTrue(selected_test_mock.call_args.kwargs["dry_run"])
            write_report_mock.assert_not_called()

    def test_main_all_dry_run_calls_selected_test_eval(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="stage3-main-all-dry-run-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module = load_stage3_module(tmp_path)
            output = io.StringIO()
            argv = [
                "run_pusht_stage3.py",
                "--mode",
                "all",
                "--ids",
                "baseline",
                "--train-seeds",
                "3072",
                "--epochs",
                "5",
                "50",
                "--dry-run",
            ]

            with mock.patch.object(module, "ensure_manifests", return_value={"val_small": Path("val.json"), "test": Path("test.json")}), mock.patch.object(
                module,
                "train_variant",
                return_value=True,
            ), mock.patch.object(
                module,
                "eval_variant",
                return_value=True,
            ), mock.patch.object(
                module,
                "run_selected_test_evals",
                return_value=True,
            ) as selected_test_mock, mock.patch.object(module, "write_report") as write_report_mock, mock.patch.object(
                sys,
                "argv",
                argv,
            ), mock.patch(
                "sys.stdout",
                output,
            ):
                module.main()

            selected_test_mock.assert_called_once()
            self.assertTrue(selected_test_mock.call_args.kwargs["dry_run"])
            write_report_mock.assert_not_called()
            self.assertIn("DRY RUN: would write Stage 3 report", output.getvalue())


if __name__ == "__main__":
    unittest.main()
