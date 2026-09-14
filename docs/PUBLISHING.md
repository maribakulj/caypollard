# Publishing to GitHub

The repository can be published with GitHub CLI after the archive is extracted.

## Public repository

```bash
cd caypollard
./scripts/publish_to_github.sh caypollard public
```

## Private repository

```bash
cd caypollard
./scripts/publish_to_github.sh caypollard private
```

The script:

1. verifies `git` and `gh` are installed;
2. verifies GitHub CLI authentication;
3. initialises a local Git repository if the archive did not preserve `.git`;
4. creates the initial commit if needed;
5. creates the GitHub repository;
6. adds `origin` and pushes `main`.

The project source code is MIT licensed. External heritage data remain under their source licences and rights statements.
