# Pull-request integration audit

Date: 2026-11-09. Audited against unified `main` at the commit that records the final Termux/Alpine suites.

## Result

- **58 PRs examined**: 56 merged, 2 closed without merge.
- **56/56 merged PR merge commits are ancestors of `main`.**
- **422/422 paths changed by merged PRs are present or absent exactly as the PR intended**; no file was silently dropped.
- **All added tests are present in the unified tree and the full suite passes** (Termux 1,613 tests, 1 skipped; Alpine 1,613 tests, 8 skipped).
- **2 closed PRs are superseded by merged PR #4 and current tests**, not lost work. Their unique validation and non-promotion behavior is present in main.
- The only literal-line mismatches are later updates to the same CI workflow pins/steps (PRs #30/#35/#38/#46) and old release metadata (PR #7); each current equivalent is present.

## Per-PR result

| PR | State | Merge commit | Changed files | Test files | Code content | Finding |
|---:|---|---:|---:|---:|---:|---|
| #1 | merged | `4c4a4c78` | 21 | 6 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #2 | closed | — | 4 | 1 | superseded | CLOSED unmerged alternative head-evaluation hardening. Superseded by merged PR #4 and current tests; main has stricter validation and non-promotion enforcement. 6/8 of its old tests pass here; one expects validation order source_digest-before-resolution, one uses a temporary-file path assumption. |
| #3 | closed | — | 4 | 1 | superseded | CLOSED unmerged alternative final head-evaluation variant. Superseded by merged PR #4 and current tests; main has numeric N16REL validation, dataset digest binding and non-promotion enforcement. 6/8 old tests pass; two fail on temporary path/secret-environment assumptions only. |
| #4 | merged | `4740b9a5` | 8 | 3 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #5 | merged | `4bc4d278` | 45 | 11 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #6 | merged | `645090b6` | 20 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #7 | merged | `c31b97b5` | 11 | 1 | 64% superseded | Release v1.2.2 metadata/demo lines superseded by later releases; current demo covers the final 86-tool surface. |
| #8 | merged | `314219d6` | 23 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #9 | merged | `ac1dd2c1` | 27 | 3 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #10 | merged | `35519cec` | 27 | 4 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #11 | merged | `5cf2a47c` | 23 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #12 | merged | `9287f801` | 21 | 3 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #13 | merged | `f78f9a4e` | 6 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #14 | merged | `21b975e3` | 9 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #15 | merged | `42b0e936` | 5 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #16 | merged | `43ff7958` | 6 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #17 | merged | `587de9cd` | 12 | 3 | integrated | Round 042 branch merged; subsequent local commits extended it with evidence lanes and TRIGON-9 artifacts. |
| #18 | merged | `b697416d` | 4 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #19 | merged | `e72ca6cf` | 5 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #20 | merged | `f9207457` | 4 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #21 | merged | `cd7c45bf` | 3 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #22 | merged | `c861778f` | 1 | 0 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #23 | merged | `8bb296db` | 12 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #24 | merged | `b9aeb75b` | 2 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #25 | merged | `9a95c632` | 3 | 0 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #26 | merged | `d8cf8989` | 3 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #27 | merged | `a0dc245a` | 6 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #28 | merged | `4fd4a53a` | 8 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #29 | merged | `ce9a7d53` | 3 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #30 | merged | `2b3b58e8` | 1 | 0 | 0% superseded | Pinned router ref superseded by later PRs; the current workflow keeps single-chart pinning. |
| #31 | merged | `0594f3f4` | 4 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #32 | merged | `82d3ef03` | 1 | 0 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #33 | merged | `76c2e4cb` | 2 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #34 | merged | `fb0bb48e` | 3 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #35 | merged | `6a515701` | 1 | 0 | 0% superseded | Older router/NaN0S pins and E2E step superseded by later workflow updates; current pinned E2E remains. |
| #36 | merged | `e7b2edad` | 2 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #37 | merged | `290f7139` | 7 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #38 | merged | `8ea57e2d` | 1 | 0 | 0% superseded | Older checks and step name superseded by later multi-head workflow updates. |
| #39 | merged | `f49079e8` | 2 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #40 | merged | `876fd8f4` | 9 | 3 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #41 | merged | `72f413c1` | 4 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #42 | merged | `1d090ba7` | 2 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #43 | merged | `a191b675` | 5 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #44 | merged | `2d6ed9f5` | 6 | 2 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #45 | merged | `0f40d5c6` | 5 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #46 | merged | `941e4708` | 1 | 0 | 0% superseded | Router pin and step name superseded by later four-class pin (#53 and successors); current workflow retains the gate. |
| #47 | merged | `03826d8c` | 2 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #48 | merged | `bd06cae1` | 6 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #49 | merged | `a3ab53d1` | 3 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #50 | merged | `a7331f03` | 6 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #51 | merged | `f90d4e4e` | 3 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #52 | merged | `4e04c20e` | 1 | 0 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #53 | merged | `12ed813b` | 1 | 0 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #54 | merged | `604ea93a` | 1 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #55 | merged | `e070eedb` | 3 | 0 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #56 | merged | `60ab0f64` | 4 | 1 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #57 | merged | `1d2e05c2` | 4 | 0 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |
| #58 | merged | `d7a27e9d` | 14 | 4 | integrated | Merge commit in `main`; changed paths retained; tests included in unified suite. |

## How this was checked

1. Enumerated all PRs with `gh pr list --state all`.
2. For every merged PR, verified its merge commit is an ancestor of unified `main` with `git merge-base --is-ancestor`.
3. Diffed each merge commit against its first parent and checked every changed path is present (added/modified) or absent (deleted) in `main`.
4. For code files, checked the literal added lines still appear in `main`; flagged later edits. The five below-80% cases are the CI pin/step updates and the old 1.2.2 release metadata listed above.
5. Fetched the closed PR heads from `refs/pull/N/head`; verified their feature set is present in the merged PR #4 implementation and current `tests/test_head_cortex_evaluation.py`.
6. Ran the full unified suite on Termux and Alpine; all added test modules are collected.

This audit verifies history, file retention, feature retention and tests. It does not claim that each PR's own CI matrix was rerun in isolation; the unified suite is the current proof.

