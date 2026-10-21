# Candidate packaging and publication

The immutable historical [v0.1.0a1 release](https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1) and [GitHub Pages site](https://zevin17.github.io/deepinfer-mac/) are published. Working0.1.0a2 is an experimental update in preparation, not a published or jointly accepted release.

Build candidates into an isolated directory so publisheda1 bytes remain unchanged:

```sh
python3 site/build.py
python3 site/check.py
uv build --out-dir dist/v0.1.0a2
python3 scripts/verify_artifacts.py --dist dist/v0.1.0a2
```

Run portable checks, actual Metal controls and permitted report verification on the final source. Inspect source/wheel/site resources, locks, licenses and exclusions. Restricted public benchmark tasks/fixtures/generations stay in results/quality-cache and never enter the static site or package. Record candidate hashes and source state. Until final comparison completes, call this an experimental update and retain the open competitive goal.

Pages builds site/dist through the existing workflow and deploys relative links under /deepinfer-mac/. Local website validation does not establish that a candidate has been deployed. No community announcement is sent by this task.

Draft: “DeepInfer-Mac adds an executable DeepInfer-derived scheduler and hybrid-state recovery controls, plus a real9B baseline/sampling demo. Original C++ replay and model outputs are inspectable. Fair competitive advantage remains unproven; model quality and performance comparisons remain separate.”

中文：a1历史发布保持不变，a2先完成可复查产物。当前可以报告方法/正确性进展，不能宣传竞争加速或联合验收完成。
