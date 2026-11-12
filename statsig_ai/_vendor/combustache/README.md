# combustache (vendored)

- Upstream: https://github.com/sakhezech/combustache
- Version: 3.0.1, from the PyPI wheel `combustache-3.0.1-py3-none-any.whl`
- License: MIT, see `LICENSE`

The files are unmodified. Its imports were already relative, so none needed
rewriting for the `statsig_ai._vendor.combustache` path.

Changes from upstream:

- Dropped `__main__.py`, the command-line interface. The package docstring in
  `__init__.py` still mentions the CLI.
- Added this README.

To update, replace the `.py` files and `LICENSE` with the new release's, keep
`__main__.py` out, and update the version above.
