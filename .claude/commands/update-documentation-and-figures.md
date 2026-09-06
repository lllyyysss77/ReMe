---
name: update-documentation-and-figures
description: Workflow command scaffold for update-documentation-and-figures in ReMe.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /update-documentation-and-figures

Use this workflow when working on **update-documentation-and-figures** in `ReMe`.

## Goal

Enhances documentation content, layout, and styling, including updating README files and associated SVG/GIF figures for both English and Chinese versions.

## Common Files

- `README.md`
- `README_ZH.md`
- `docs/figure/*.svg`
- `docs/figure/*.gif`
- `docs/figure/*.mp4`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit README.md and README_ZH.md to update content, layout, and formatting.
- Modify or add SVG/GIF files under docs/figure/ to update diagrams or demo assets.
- Synchronize changes across both English and Chinese documentation.
- Adjust table/image dimensions and styling for consistency.
- Optionally remove outdated media assets from docs/figure/.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.