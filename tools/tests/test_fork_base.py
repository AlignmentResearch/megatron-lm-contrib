#!/usr/bin/env python3
# Copyright (c) 2026, FAR.AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused tests for root fork identity and PR-to-target identity stability."""

import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "fork_base.py"
SPEC = importlib.util.spec_from_file_location("fork_base", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fork_base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fork_base)


class TestForkBaseIdentity(unittest.TestCase):
    def test_only_internal_fork_workflow_remains(self) -> None:
        root = Path(__file__).parents[2]
        self._assert_nvidia_automation_pruned(root)

    def test_sync_prunes_reintroduced_upstream_automation(self) -> None:
        source_root = Path(__file__).parents[2]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            upstream = temp / "upstream"
            fork = temp / "fork"
            key = temp / "signing-key"

            self._git("init", "-b", "main", str(upstream), cwd=temp)
            self._configure_repo(upstream)
            self._write(upstream / ".github/workflows/nvidia.yml", "name: NVIDIA\n")
            self._write(upstream / ".github/CODEOWNERS", "* @NVIDIA/mcore\n")
            self._write(upstream / ".github/copy-pr-bot.yaml", "enabled: true\n")
            self._write(upstream / "upstream.txt", "base\n")
            self._git("add", ".", cwd=upstream)
            self._git("commit", "-m", "upstream base", cwd=upstream)
            base = self._git("rev-parse", "HEAD", cwd=upstream).stdout.strip()

            self._git("clone", str(upstream), str(fork), cwd=temp)
            self._configure_repo(fork)
            self._git("remote", "add", "upstream", str(upstream), cwd=fork)
            (fork / "tools").mkdir()
            shutil.copy2(source_root / "tools/fork_base.py", fork / "tools/fork_base.py")
            shutil.copy2(source_root / "tools/sync_upstream.sh", fork / "tools/sync_upstream.sh")
            shutil.copytree(
                source_root / ".github/workflows", fork / ".github/workflows", dirs_exist_ok=True
            )
            for path in (fork / ".github/workflows").iterdir():
                if path.name != "fork-base.yml":
                    path.unlink()
            (fork / ".github/CODEOWNERS").unlink()
            (fork / ".github/copy-pr-bot.yaml").unlink()
            manifest = {
                "schema": 1,
                "fork_repo": str(upstream),
                "upstream_repo": str(upstream),
                "upstream_branch": "main",
                "upstream_base": base,
            }
            self._write(fork / ".fork-base.json", json.dumps(manifest, indent=2) + "\n")
            self._git("add", ".", cwd=fork)
            self._git("commit", "-m", "fork scaffolding", cwd=fork)

            self._write(upstream / ".github/workflows/new.yml", "name: New NVIDIA workflow\n")
            self._write(upstream / ".github/CODEOWNERS", "* @NVIDIA/new-team\n")
            self._write(upstream / ".github/copy-pr-bot.yaml", "enabled: false\n")
            self._write(upstream / "upstream.txt", "updated\n")
            self._git("add", ".", cwd=upstream)
            self._git("commit", "-m", "upstream update", cwd=upstream)

            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True
            )
            self._git("config", "gpg.format", "ssh", cwd=fork)
            self._git("config", "user.signingkey", str(key), cwd=fork)
            subprocess.run(
                ["bash", "tools/sync_upstream.sh"],
                cwd=fork,
                check=True,
                capture_output=True,
                text=True,
            )

            self._assert_nvidia_automation_pruned(fork)
            self.assertEqual(self._git("status", "--porcelain", cwd=fork).stdout, "")

            for commit in self._git(
                "rev-list", "--max-count=2", "HEAD", cwd=fork
            ).stdout.splitlines():
                raw_commit = self._git("cat-file", "commit", commit, cwd=fork).stdout
                self.assertIn("gpgsig -----BEGIN SSH SIGNATURE-----", raw_commit)
                self.assertIn("Signed-off-by: Test User <test@example.com>", raw_commit)

            self._write(fork / ".github/scripts/helper.py", "fork implementation\n")
            self._git("add", ".github/scripts/helper.py", cwd=fork)
            self._git("commit", "-m", "fork helper change", cwd=fork)
            self._write(upstream / ".github/scripts/helper.py", "upstream implementation\n")
            self._write(upstream / ".github/workflows/another.yml", "name: More NVIDIA CI\n")
            self._write(upstream / ".github/CODEOWNERS", "* @NVIDIA/another-team\n")
            self._write(upstream / ".github/copy-pr-bot.yaml", "enabled: true\n")
            self._git("add", ".", cwd=upstream)
            self._git("commit", "-m", "upstream helper conflict", cwd=upstream)

            conflicted = subprocess.run(
                ["bash", "tools/sync_upstream.sh"],
                cwd=fork,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(conflicted.returncode, 0)
            unmerged = self._git(
                "diff", "--name-only", "--diff-filter=U", cwd=fork
            ).stdout.splitlines()
            self.assertEqual(unmerged, [".github/scripts/helper.py"])
            self._assert_nvidia_automation_pruned(fork)

    def _assert_nvidia_automation_pruned(self, root: Path) -> None:
        workflows = sorted(
            path.relative_to(root).as_posix()
            for path in (root / ".github" / "workflows").rglob("*")
            if path.is_file()
        )
        self.assertEqual(workflows, [".github/workflows/fork-base.yml"])
        self.assertFalse((root / ".github" / "CODEOWNERS").exists())
        self.assertFalse((root / ".github" / "copy-pr-bot.yaml").exists())

    @staticmethod
    def _configure_repo(repo: Path) -> None:
        TestForkBaseIdentity._git("config", "user.name", "Test User", cwd=repo)
        TestForkBaseIdentity._git("config", "user.email", "test@example.com", cwd=repo)

    @staticmethod
    def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_root_fork_must_match_origin(self) -> None:
        manifest = {"fork_repo": "https://github.com/AlignmentResearch/Megatron-LM"}
        with patch.object(
            fork_base, "_git", return_value="https://github.com/example/other-fork.git"
        ):
            self.assertEqual(
                fork_base.check_root_identity(Path("/repo"), manifest),
                [
                    "fork_repo https://github.com/AlignmentResearch/Megatron-LM "
                    "!= origin https://github.com/example/other-fork.git"
                ],
            )

    def test_pull_request_cannot_change_manifest_identity(self) -> None:
        previous = {
            "fork_repo": "https://github.com/AlignmentResearch/Megatron-LM",
            "upstream_repo": "https://github.com/NVIDIA/Megatron-LM",
            "upstream_branch": "main",
            "upstream_base": "a" * 40,
        }
        current = {**previous, "upstream_repo": "https://github.com/example/fork"}
        with patch.object(fork_base, "_git", return_value=json.dumps(previous)):
            problems, notes = fork_base.check_forward(
                Path("/repo"), current, previous["upstream_base"], "origin/farai/main"
            )

        self.assertEqual(notes, [])
        self.assertEqual(
            problems,
            [
                "manifest identity changed for upstream_repo: origin/farai/main "
                "records https://github.com/NVIDIA/Megatron-LM, current records "
                "https://github.com/example/fork"
            ],
        )

    def test_a_rename_of_this_repository_is_allowed(self) -> None:
        """The manifest may follow the repository's own rename, and only that.

        `fork_repo` has to equal `origin` (test_root_fork_must_match_origin), so the old name
        fails root identity and the new one would fail the identity-stability rule: without this
        carve-out a rename cannot be recorded through a pull request at all.
        """
        previous = {
            "fork_repo": "https://github.com/AlignmentResearch/Megatron-LM",
            "upstream_repo": "https://github.com/NVIDIA/Megatron-LM",
            "upstream_branch": "main",
            "upstream_base": "a" * 40,
        }
        renamed = {
            **previous,
            "fork_repo": "https://github.com/AlignmentResearch/megatron-lm-contrib",
        }

        def git(*args: str, **kwargs: object) -> str:
            if args[:2] == ("remote", "get-url"):
                return "https://github.com/AlignmentResearch/megatron-lm-contrib.git"
            return json.dumps(previous)

        with patch.object(fork_base, "_git", side_effect=git):
            problems, notes = fork_base.check_forward(
                Path("/repo"), renamed, previous["upstream_base"], "origin/farai/main"
            )

        self.assertEqual((problems, notes), ([], []))

    def test_a_fork_repo_that_is_not_this_repository_is_still_refused(self) -> None:
        """The carve-out is a rename, not a free hand: a new name that origin does not answer to
        is still a different fork, and two forks sharing an upstream base carry different patches.
        """
        previous = {
            "fork_repo": "https://github.com/AlignmentResearch/Megatron-LM",
            "upstream_repo": "https://github.com/NVIDIA/Megatron-LM",
            "upstream_branch": "main",
            "upstream_base": "a" * 40,
        }
        elsewhere = {**previous, "fork_repo": "https://github.com/example/other-fork"}

        def git(*args: str, **kwargs: object) -> str:
            if args[:2] == ("remote", "get-url"):
                return "https://github.com/AlignmentResearch/megatron-lm-contrib.git"
            return json.dumps(previous)

        with patch.object(fork_base, "_git", side_effect=git):
            problems, _ = fork_base.check_forward(
                Path("/repo"), elsewhere, previous["upstream_base"], "origin/farai/main"
            )

        self.assertEqual(
            problems,
            [
                "manifest identity changed for fork_repo: origin/farai/main "
                "records https://github.com/AlignmentResearch/Megatron-LM, current records "
                "https://github.com/example/other-fork"
            ],
        )

    def test_missing_target_manifest_is_a_bootstrap_note(self) -> None:
        missing = subprocess.CalledProcessError(128, ["git", "show"])
        with patch.object(fork_base, "_git", side_effect=missing):
            problems, notes = fork_base.check_forward(
                Path("/repo"), {}, "a" * 40, "origin/farai/main"
            )

        self.assertEqual(problems, [])
        self.assertEqual(
            notes,
            ["origin/farai/main has no .fork-base.json yet — forward-only check not applicable"],
        )


if __name__ == "__main__":
    unittest.main()
