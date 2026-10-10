# Repository Instructions

## Development artifacts

- This policy applies to ordinary development sessions, skills, and workflows.
- Store task-generated intermediate artifacts under
  `<project-root>/devlocal/<feat-name>/`.
- `project-root` is the root of the project or Git worktree being modified,
  not the session's parent directory or the skill's installation directory.
- Use a stable, concise kebab-case feature name. Reuse the feature directory
  across related sessions; use separate run subdirectories for repeated
  experiments to preserve existing evidence.
- Intermediate artifacts include analysis reports, plans, experimental code,
  one-off scripts, debug probes, logs, benchmark results, traces, screenshots,
  generated captures, and agent state. Create subdirectories such as `reports/`,
  `scripts/`, `logs/`, and `results/` only as needed.
- Do not scatter these artifacts across `doc/`, `docs/`, `scripts/`, `tests/`,
  or the project root. Skill and workflow default output paths must follow
  this policy. An explicit user-specified output location takes precedence.
- Keep maintained product code, tests, reusable tools, and formal documentation
  in their established project locations. When promoting an intermediate
  artifact into a maintained deliverable, curate it and update its references.
- `devlocal/` is ignored by Git. Do not force-add its contents; keep the review
  diff limited to maintained deliverables. Verify ignore rules and the diff
  before preparing a review.
- Reuse existing external checkpoints, reference repositories, and datasets in
  place; record their paths and revisions in the feature directory rather than
  copying or moving them merely to satisfy this layout. Keep credentials out
  of generated artifacts.
- Existing build and dependency caches may retain their tool-managed locations.
  Use explicit output-directory options for task-specific reports and runs.

## Model integration and module ownership

When adding a family, changing precision/preparation, or moving model code, read
[Model Layer Architecture](doc/model-layer-architecture.md#current-module-names-and-responsibilities)
and [Adding a New Model](doc/adding-a-new-model.md) from the checkout being modified.
Use [model-port-workflow](skills/model-port-workflow/SKILL.md) for a full port.
Keep module/capability docs and relevant skills in the same change; historical
Session/Network proposals are not current integration APIs.

# ApxInf project rules

## Scope

These rules apply to new first-party work in this repository.
Keep existing dependency and vendor records intact.

## Independent implementation

- Write all new implementation code by hand.
- Write all new tests and fixtures by hand.
- Use external repositories to study behavior, interfaces, and design choices.
- Do not copy, translate, port, or lightly rewrite external implementation code.
- Do not copy external tests, fixtures, or code examples into this repository.
- Do not use code generators or imported project templates for new first-party code.
- Call existing dependencies through their public APIs when the design permits this use.
- Record each new dependency and its role in the design.
- Keep reference notes separate from implementation instructions.
- Do not describe this process as a clean-room process.

An installed authoring skill is a development tool.
Its installation does not make its source part of ApxInf.

## Design and interfaces

Read [the serving design](doc/serving-system-design-20261009.md) before serving work.
Read [the contract](doc/serving/contracts-v0.1.md) before an interface change.
Read [the implementation plan](doc/serving/implementation-plan.md) before a new stage.

The contract defines serving terms, fields, states, and errors.
The contract takes precedence over interface sketches in the design.
The existing v1 protocols retain their existing behavior.

- Define each interface before its implementation.
- Keep one term for each concept.
- Write Rust and Python types from the same contract.
- Check both implementations with the same original fixture corpus.
- Change the contract, types, validators, and fixtures together.
- Keep capability limits explicit.
- Do not silently change a provider, precision profile, or model revision.
- Complete each stage gate before dependent work.

## Technical writing

Use ASD-STE100 Issue 9 principles for normative English documents.
Use the installed `asd-ste100` skill when it is available.
Use active voice and short sentences.
Define each technical term once.
Keep instructions within 20 words.
Keep descriptions within 25 words.
Keep each paragraph to one topic and six sentences or fewer.

Keep Chinese explanations for discussion with the user.
Do not describe Chinese prose as compliant STE English.
Report structural checks separately from full dictionary review.
Do not claim certification from a skill or a linter.
