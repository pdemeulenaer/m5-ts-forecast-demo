# Publish the documentation

The documentation is built with MkDocs and published to GitHub Pages by
[`.github/workflows/docs.yml`](https://github.com/pdemeulenaer/m5-ts-forecast-demo/blob/main/.github/workflows/docs.yml).
Pull requests build the site to catch broken links and MkDocs errors. A push to
`main` builds and deploys it; you can also start a deployment from the Actions tab.

## Enable GitHub Pages

Do this once in the repository settings:

1. Open **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Push the workflow to `main`, or run **Documentation** from the **Actions** tab.

The workflow installs the documentation dependencies from `uv.lock`, runs
`mkdocs build --strict`, then deploys the generated `site/` directory. It uses the
repository's `GITHUB_TOKEN`; no deploy key or personal access token is needed.

After the first successful run, the site is available at
<https://pdemeulenaer.github.io/m5-ts-forecast-demo/>. The Actions run also shows
the deployed URL. Future pushes to `main` that change documentation, MkDocs
configuration, Python API source, or documentation dependencies publish an update.

## Build locally

The same strict build runs from the repository root:

```bash
uv sync --extra cpu --group docs
uv run --extra cpu --group docs mkdocs build --strict
```

For a live preview, use `make docs` and open <http://127.0.0.1:8000>.
