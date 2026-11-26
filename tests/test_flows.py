"""Every `winterload` command in the Kestra flows must parse with the real CLI parser."""

import re
import shlex
from pathlib import Path

import pytest
import yaml

from winterload.pipeline import STAGE_NAMES, build_parser

FLOWS = sorted(Path(__file__).parent.parent.joinpath("docker/kestra/files/flows").glob("*.yml"))
PEBBLE = re.compile(r"(--[\w-]+\s+)?(\{\{.*?\}\})")
NEEDS_ANY_STRING = {"--scheduled-at", "--kestra-execution-id", "--tuning-run-id"}


def _stand_in(match: re.Match) -> str:
    flag = (match.group(1) or "").strip()
    value = "1" if flag == "--trials" else "x" if flag in NEEDS_ANY_STRING else "2026-11-26"
    return f"{match.group(1) or ''}{value}"


def _commands(flow: Path) -> list[str]:
    tasks = yaml.safe_load(flow.read_text()).get("tasks", [])
    lines = [c for task in tasks for c in task.get("commands", [])]
    return [line.strip() for line in lines if line.strip().startswith("winterload ")]


CASES = [pytest.param(flow, cmd, id=f"{flow.name}::{cmd}") for flow in FLOWS for cmd in _commands(flow)]


def test_flows_found():
    assert FLOWS


@pytest.mark.parametrize("flow", FLOWS, ids=lambda f: f.name)
def test_flow_has_winterload_command(flow):
    assert _commands(flow)


@pytest.mark.parametrize(("flow", "command"), CASES)
def test_command_parses(flow, command):
    args = shlex.split(PEBBLE.sub(_stand_in, command))
    parsed = build_parser().parse_args(args[1:])
    assert set(parsed.pipeline or []) <= set(STAGE_NAMES)
