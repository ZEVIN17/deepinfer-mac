"""Private fixed-history logits diagnosis; no runtime edits or performance claims."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def main():
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator, GenerationBatch

    root = Path(__file__).resolve().parents[2]
    dest = root / "artifacts/phase3-continuous/logits-diagnosis"
    dest.mkdir(exist_ok=False)
    source = root / "artifacts/phase3-continuous/interactive-fresh.json"
    raw = json.loads(source.read_text())
    cfg = json.loads((root / "examples/models/smollm2-135m-8bit.local.json").read_text())
    weights = Path(cfg["source"])
    files = sorted(weights.glob("*.safetensors"))
    assert files and sum(p.stat().st_size for p in files) < 200_000_000
    import psutil

    assert psutil.virtual_memory().available > 2 * 1024**3
    swap_before = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    model, tok = load(str(weights))
    context = {}
    original = GenerationBatch._step

    def step(batch):
        context["batch"] = batch
        try:
            return original(batch)
        finally:
            context.pop("batch", None)

    GenerationBatch._step = step
    records = []
    arrays = {}

    class Probe:
        def __getattr__(self, name):
            return getattr(model, name)

        def __call__(self, inputs, cache=None, **kwargs):
            b = context.get("batch")
            before = cache_meta(cache)
            logits = model(inputs, cache=cache, **kwargs)
            if b is not None and target_uid in b.uids:
                row = b.uids.index(target_uid)
                index = len(current)
                vector = np.array(logits[row, -1, :].astype(mx.float32))
                order = np.argsort(vector)[-2:][::-1]
                current.append(
                    {
                        "index": index,
                        "input_token": int(inputs[row, -1].item()),
                        "shape": list(inputs.shape),
                        "row": row,
                        "uids": list(b.uids),
                        "cache_before": before,
                        "cache_after": cache_meta(cache),
                        "top2": order.tolist(),
                        "top2_logits": vector[order].tolist(),
                        "margin": float(vector[order[0]] - vector[order[1]]),
                    }
                )
                vectors.append(vector)
                if force and index < len(reference):
                    logits[row, -1, reference[index]] = mx.max(logits[row, -1, :]) + 100
            return logits

    def cache_meta(cache):
        c = cache[0]
        out = {"type": type(c).__name__}
        for key in ["offset", "_offset", "left_padding", "_left_padding", "lengths"]:
            value = getattr(c, key, None)
            if value is not None:
                out[key] = value.tolist() if hasattr(value, "tolist") else value
        for key in ["keys", "values"]:
            value = getattr(c, key, None)
            if value is not None:
                out[key + "_shape"] = list(value.shape)
                out[key + "_dtype"] = str(value.dtype)
        return out

    reference = raw["native"][2]["output_token_ids"]
    for cap, force in [(1, False), (4, False), (1, True), (4, True)]:
        name = f"b{cap}-" + ("forced" if force else "natural")
        indices = [2] if cap == 1 else [0, 1, 2, 3]
        current, vectors = [], []
        # insert_segments allocates UIDs from zero and runs no model calls.
        target_uid = indices.index(2)
        g = BatchGenerator(
            Probe(),
            stop_tokens=[[t] for t in tok.eos_token_ids],
            completion_batch_size=cap,
            prefill_batch_size=1,
            prefill_step_size=32,
        )
        ids = g.insert_segments(
            [[raw["native"][i]["prompt_token_ids"]] for i in indices],
            max_tokens=[24 if i == 1 else 48 for i in indices],
        )
        assert ids[indices.index(2)] == target_uid
        outputs = {uid: [] for uid in ids}
        done = set()
        while len(done) < len(ids):
            _, responses = g.next()
            for response in responses:
                outputs[response.uid].append(response.token)
                if response.finish_reason:
                    done.add(response.uid)
        g.close()
        arrays[name] = np.stack(vectors)
        records.append(
            {
                "name": name,
                "calls": current,
                "outputs": outputs,
                "target_matches_serial": outputs[target_uid] == reference,
            }
        )
    np.savez_compressed(dest / "logits.npz", **arrays)
    comparisons = []
    for mode in ["natural", "forced"]:
        a, b = arrays[f"b1-{mode}"], arrays[f"b4-{mode}"]
        n = min(len(a), len(b), 48)
        differences = np.max(np.abs(a[:n] - b[:n]), axis=1)
        argdiff = np.flatnonzero(a[:n].argmax(1) != b[:n].argmax(1))
        comparisons.append(
            {
                "mode": mode,
                "compared": n,
                "first_nonidentical_logits": next((i for i, d in enumerate(differences) if d != 0), None),
                "first_argmax_difference": int(argdiff[0]) if len(argdiff) else None,
                "max_abs_logit_differences": differences.tolist(),
            }
        )
    payload = {
        "records": records,
        "comparisons": comparisons,
        "versions": {k: importlib.metadata.version(k) for k in ["mlx", "mlx-lm", "numpy"]},
        "weight_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metal_peak_bytes": mx.get_peak_memory(),
        "swap_before": swap_before,
        "swap_after": subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip(),
        "performance_claim": False,
    }
    (dest / "raw.json").write_text(json.dumps(payload, indent=2))
    del g
    mx.clear_cache()
    print(json.dumps({"comparisons": comparisons, "peak": payload["metal_peak_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
