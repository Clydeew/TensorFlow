# Publish when you choose

This is a local Git repository. The assistant did not create a GitHub repository, set a remote or push. Review `git status` and the files first. No license is assumed: select your intended license before public distribution; until then no additional license grant is made.

From this directory, after creating an empty GitHub repository yourself:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin codex/initial-app
```

These commands are instructions only and were not run by the assistant. Add real datasets, credentials, model binaries or generated runs only after deliberate review; the defaults ignore them. `models/` may contain separately licensed weights and should stay outside Git. Add a project license and appropriate notices before release. If you prefer a main branch, rename locally with `git branch -m main` and push that name instead.
