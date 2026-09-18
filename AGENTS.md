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
