```markdown
# ReMe Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill introduces the core development conventions and workflows used in the ReMe Python codebase. It covers file organization, code style, commit practices, and documentation update workflows, equipping contributors to maintain consistency and quality throughout the project.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python file names.
  - Example: `data_loader.py`, `user_profile.py`

### Import Style
- Prefer **relative imports** within the package.
  - Example:
    ```python
    from .utils import parse_config
    from .models import User
    ```

### Export Style
- Use **named exports** by explicitly listing public objects in `__all__`.
  - Example:
    ```python
    __all__ = ['User', 'parse_config']
    ```

### Commit Messages
- Follow **Conventional Commits** with these prefixes:
  - `feat`: New features
  - `fix`: Bug fixes
  - `docs`: Documentation changes
  - `chore`: Maintenance tasks
- Keep commit messages concise (~61 characters on average).
  - Example: `feat: add user authentication middleware`

## Workflows

### Update Documentation and Figures
**Trigger:** When you want to improve or expand the project's documentation, especially visual elements or feature explanations.  
**Command:** `/update-docs`

1. Edit `README.md` and `README_ZH.md` to update content, layout, or formatting.
2. Modify or add SVG/GIF files under `docs/figure/` to update diagrams or demo assets.
3. Synchronize changes across both English and Chinese documentation.
4. Adjust table/image dimensions and styling for consistency.
5. Optionally remove outdated media assets from `docs/figure/`.

**Example:**
```bash
# Edit documentation
vim README.md
vim README_ZH.md

# Add or update a figure
cp new_diagram.svg docs/figure/

# Remove outdated assets
rm docs/figure/old_demo.gif

# Commit changes
git add README.md README_ZH.md docs/figure/
git commit -m "docs: update figures and synchronize documentation"
```

## Testing Patterns

- **Framework:** Unknown (no standard detected)
- **Test File Pattern:** `*.test.ts`
  - Indicates that some tests may be written in TypeScript, even though the main codebase is Python.
- **Best Practice:** Place test files alongside or within a dedicated test directory, following the `snake_case` naming convention where possible.

## Commands

| Command       | Purpose                                                      |
|---------------|--------------------------------------------------------------|
| /update-docs  | Update documentation content, layout, and associated figures. |

```