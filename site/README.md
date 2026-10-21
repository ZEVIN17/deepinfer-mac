# Static showcase

Dependency-free HTML/CSS generated with the Python standard library. No inference endpoint, visitor prompt collection, remote fonts or automatic localhost access.

```sh
python3 site/build.py
python3 site/check.py
python3 -m http.server 8088 --directory site/dist
```

For exact GitHub Pages path checking, serve a temporary directory containing `deepinfer-mac -> site/dist` and open `/deepinfer-mac/`. All resources and internal links are relative. The workflow uploads **the contents of site/dist**, which is the complete deployable artifact.

To publish reviewed real benchmark data, use `python3 site/build.py --report benchmarks/results/<run>/summary.json`; preserve that report in `site/data/benchmark.json` for CI rebuilds. Only schema-2 reports with full samples and environment are accepted. Missing data displays an explicit unmeasured state; no synthetic data is generated.

The local candidate site includes English/Chinese entry pages, architecture, evidence, contribution, compatibility and a current bilingual Phase2 evidence page and a historical27B page, plus complete downloadable source documentation/licenses/locks. `site/dist` is generated and excluded from Git. GitHub Pages source must be set to GitHub Actions before `.github/workflows/pages.yml` can deploy.

Current Phase2 data use an explicit permitted-report allowlist in build.py; restricted public dataset raw content stays outside the site. The old alpha benchmark.json/source archive are retained with historical version/model labels. Local build success does not mean the candidate has been deployed.
