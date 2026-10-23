# MP28 validation — 2026-10-23

Implementation and teaching criteria in the [brief](README.md) are complete; learner mastery unassessed.

Executed the three README commands using Python3.12.14 / NumPy2.5.4, CPU. All five unit tests passed: independent two-scalar-head hand solution; single-head reference equivalence; zero output projection/parameter counts; permutation equivariance and immutability; invalid dimensions/projections. The experiment produced six finite conditions using one checksummed valid training crop and a synthetic fixture; no validation/test data, optimization or acquisition.

All seven notebook code cells passed in a real Jupyter kernel, exit0; [execution record](../../reports/notebook-execution.json). Two plots were captured for visual review. The pre-existing local kernel TCP transport warning appeared, with no execution error.

The [measured report](../../reports/mp28-multihead.json) records seed, software, source hash/license/crop and all maps. Arrays have shapes[heads,16,16]; projection counts stay64 and float64 map sizes increase2048→4096→8192 bytes. These are actual array sizes, not measured peak RAM. Different random maps do not establish task benefit. No positions, biases, dropout, masks, trained model or SR metric is included; these are explicitly outside this primitive's scope.
