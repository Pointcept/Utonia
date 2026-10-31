# Copyright (c) Meta Platforms, Inc. and affiliates.
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

"""Upload a Utonia checkpoint to the Hugging Face Hub in ``PyTorchModelHubMixin`` layout.

The Hub only tracks download statistics for files it knows about. For models
loaded with ``PyTorchModelHubMixin`` that means ``config.json`` and
``model.safetensors``, which this script generates from an existing ``.pth``
checkpoint via ``save_pretrained``. Only those two files are uploaded: the
existing model card (``README.md``) and the ``.pth`` files in the repo are left
untouched, so ``utonia.load(...)`` keeps working as before.

Example (requires write access to the target repo, ``hf auth login`` first):

    python tools/push_to_hub.py --name utonia --repo-id Pointcept/Utonia

Use ``--dry-run`` to only write the files to ``--output-dir`` and inspect them.
"""

import argparse
import os

import utonia


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--name",
        default="utonia",
        help="registered checkpoint name (utonia, utonia_small, utonia_tiny) or local .pth path",
    )
    parser.add_argument(
        "--repo-id",
        default="Pointcept/Utonia",
        help="Hub repo to read the .pth from and to upload the mixin files to",
    )
    parser.add_argument(
        "--revision", default=None, help="branch to push to (default: main)"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="open a pull request on the Hub repo instead of committing to the branch",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="local directory for save_pretrained (default: ./hub/<name>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only run save_pretrained locally, do not upload anything",
    )
    args = parser.parse_args()

    model = utonia.load(args.name, repo_id=args.repo_id)
    model.eval()

    output_dir = args.output_dir or os.path.join("hub", os.path.basename(args.name))
    model.save_pretrained(output_dir)
    print(f"Saved mixin files to {output_dir}: {sorted(os.listdir(output_dir))}")
    if args.dry_run:
        return

    url = model.push_to_hub(
        args.repo_id,
        branch=args.revision,
        create_pr=args.create_pr,
        # keep the hand-written model card and the existing .pth checkpoints
        allow_patterns=["config.json", "model.safetensors"],
        commit_message=f"Add PyTorchModelHubMixin weights for {args.name}",
    )
    print(f"Uploaded: {url}")


if __name__ == "__main__":
    main()
