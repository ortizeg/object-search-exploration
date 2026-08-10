---
schema_version: 1
open_count: 1
waived_count: 0
fixed_count: 0
total_count: 1
last_updated: 2026-08-10T16:17:26.711Z
---

# Broken Windows Ledger

> Cross-phase defect register. `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | quick-260810-gx5 | deviation | mkdocs.yml |  | Plan's third target docs/benchmark/floorplans-results.md does not exist; nav entry omitted. A floor-plan benchmark results page must be authored before it can be wired into the Evaluation nav. | open |  | 2026-08-10T16:17:26.711Z |  |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "quick-260810-gx5",
    "file": "mkdocs.yml",
    "line": null,
    "description": "Plan's third target docs/benchmark/floorplans-results.md does not exist; nav entry omitted. A floor-plan benchmark results page must be authored before it can be wired into the Evaluation nav.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-10T16:17:26.711Z",
    "resolved_at": null
  }
]
````
