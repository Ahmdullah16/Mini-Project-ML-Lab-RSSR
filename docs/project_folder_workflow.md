# Working inside a mini-project

Each of the 40 stage folders contains README.md, VALIDATION.md, a descriptive notebook, a tools module, run_notebook.py and a test module. The [manifest](project_folders.json) records the exact names. MP01 remains unchanged; MP02–MP40 now follow its file layout.

Install the repository development environment (`python -m pip install -e '.[dev]'`). From a project folder, run its test file with `python -m pytest -q test_...py`, then `python run_notebook.py`. Use the exact test name in that folder's README. Runners execute a real Jupyter kernel with that folder as its working directory. They record notebook source hashes and outcomes; executed copies are temporary, ignored files under data/executed_notebooks/project_folders.

From the repository root, use `python scripts/check_project_folders.py` and `python scripts/check_structure.py` for layout and links. To execute selected lessons, use `python scripts/execute_project_notebooks.py --only MP32 MP33`. Omitting --only runs all 40; this is optional and may require previously acquired training inputs. Run only changed lessons during normal development. No notebook runner creates missing research data automatically.

Edit the notebook inside the stage folder going forward. The original notebooks/ files remain historical snapshots, not a second maintained source. Shared numerical implementations remain in src/rssr; small local modules expose the relevant API, and local tests exercise that API. Standalone concepts have local implementations. This keeps each project navigable without creating divergent algorithm copies.

October 25 validation: 111 project-folder tests and 17 subtests passed, with four affine deprecation warnings. MP02–MP40 passed real kernels from their new locations; MP01 retains its original execution evidence. See [tests](../reports/project-folder-tests.json) and [notebooks](../reports/project-notebook-execution.json). Seven showcase sections and their error states passed locally, with zero network traffic on navigation.

File layout and execution are distinct from project completion. MP27–MP31 meet their implementation/teaching checklists; MP32–MP40 still need full lesson expansion and criterion-level experiments. Earlier exceptions, including MP20's independent SRCNN table, remain open. Learner mastery, public deployment verification and durable external storage are separate assessments. Frozen held-out results were not regenerated.
