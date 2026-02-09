"""Tests for eval run CLI argument parsing."""

from nothing_gpt.eval.run import DEFAULT_OUTPUT_DIR, DEFAULT_VAL_PATH, parse_args


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.output_dir == DEFAULT_OUTPUT_DIR
        assert args.val_path == DEFAULT_VAL_PATH
        assert args.max_prompts == 200
        assert args.cross_character_count == 50
        assert args.judge_count == 100
        assert args.skip_generation is False
        assert args.skip_judge is False
        assert args.wandb is False
        assert args.wandb_project == "nothing-gpt-eval"
        assert args.wandb_run_name is None

    def test_wandb_flag(self):
        args = parse_args(["--wandb"])
        assert args.wandb is True

    def test_no_wandb_flag(self):
        args = parse_args(["--wandb", "--no-wandb"])
        assert args.wandb is False

    def test_wandb_project(self):
        args = parse_args(["--wandb", "--wandb-project", "my-project"])
        assert args.wandb_project == "my-project"

    def test_wandb_run_name(self):
        args = parse_args(["--wandb", "--wandb-run-name", "test-run-1"])
        assert args.wandb_run_name == "test-run-1"

    def test_wandb_all_options(self):
        args = parse_args([
            "--wandb",
            "--wandb-project", "custom-proj",
            "--wandb-run-name", "run-42",
        ])
        assert args.wandb is True
        assert args.wandb_project == "custom-proj"
        assert args.wandb_run_name == "run-42"
