# Public diagrams

The `.mmd` files are the editable sources. The SVG files are generated from
the same source by the repository's deterministic, dependency-free renderer:

```powershell
\.venv\Scripts\python.exe scripts/render_diagrams.py --directory docs/diagrams
```

The diagrams deliberately distinguish the implemented path from staging or
future work. They do not hard-code volatile counts, benchmark results, or
production claims.
