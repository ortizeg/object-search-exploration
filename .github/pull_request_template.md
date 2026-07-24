<!--
One PR per phase checkpoint (two per phase). Fill every section — the point of this
template is that a reviewer can tell what was verified from the PR alone, without
re-running anything.
-->

## Summary

<!-- What this PR does, and why, in a few sentences. Lead with the change in behaviour. -->

## Requirements satisfied

<!--
One line per requirement ID, each stating HOW it was satisfied — concretely, not
"implemented X". If a requirement is only partially satisfied, say so and say what is left.
-->

- [ ] `REQ-ID` — how it was satisfied

## Phase success criteria verified

<!--
For each phase success criterion this PR touches, state how it was checked. Quote real
command output where it is the evidence. "Should work" is not verification.
-->

| Criterion | How it was verified |
| --- | --- |
|  |  |

## Test evidence

<!--
Paste the real output tails: `pixi run lint`, `pixi run format-check`,
`pixi run typecheck`, `pixi run test` (including the coverage line).

If this PR changes a quality gate, ALSO prove the gate fails when violated — a gate that
was never observed failing is an advisory gate.
-->

```text

```

## Notes and assumptions

<!--
Deviations from the plan and why. Assumptions a reviewer should check. Anything
deliberately deferred, with the phase that picks it up. Write "None." if there are none.
-->
