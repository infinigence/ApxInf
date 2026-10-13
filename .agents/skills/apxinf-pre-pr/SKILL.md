---
name: apxinf-pre-pr
description: Run the pre-PR check, review the diff against REVIEW.md, and assemble the evidence block for the PR template. Use before opening any pull request.
---

# Pre-PR check

Run this before opening a pull request. The point is to make sure the PR
would pass CI and to surface anything a reviewer would flag.

## Steps

1. Run `just pre-pr`. Fix anything it reports; do not open the PR red.
2. Read `REVIEW.md` and review the diff `git diff upstream/main...HEAD`
   against it. Report anything you find, most-severe first, with a
   `file:line` citation for every claim.
3. If the change touches GPU behavior and you have a device, run the
   matching `just test-gpu <family>` or `just bench <model>` recipe and
   record the output path under `devlocal/<feat>/pre-pr/`.
4. Assemble the PR-template evidence block:

   ```
   Recipe(s) run: just pre-pr [+ just test-gpu <family> | just bench <model>]
   Board / arch / commit: <board> <arch> @ <short sha>
   Skipped checks & reason: <none, or what and why>
   Docs updated in this PR: yes / not needed (why)
   AI assistance: <none | assisted (tool)>
   ```

5. Write the current HEAD SHA into `devlocal/<feat>/pre-pr.ok` so the
   `gh pr create` guard can verify it matches the branch you are about to
   push.

Never open a PR from a tree where `just pre-pr` fails, and never report
"pass" for a check you did not run.
