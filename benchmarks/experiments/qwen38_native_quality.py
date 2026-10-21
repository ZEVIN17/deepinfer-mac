"""Native MLX control for the observed 2-bit demo quality failures."""
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest

engine = InferenceEngine()
engine._load('qwen3.8-27b-2bit', False)
previous = json.loads(Path('benchmarks/reports/qwen38-27b-2bit-demo.json').read_text())
report = {'source': source_manifest(), 'cases': [], 'scope': 'native greedy control, same prompt tokens and 128-token cap; not a broad quality evaluation'}
for label, prompt in [('zh', '请用三句话解释统一内存为什么适合本地大语言模型推理。'), ('code', 'Write a Python function dedupe(items) that removes duplicates while preserving order. Return only the function.')]:
    request = GenerationRequest(profile='qwen3.8-27b-2bit', prompt=prompt)
    ids = engine._tokens(request)
    out = []
    gen = generate_step(mx.array(ids, mx.uint32), engine.model, max_tokens=128, prefill_step_size=256)
    try:
        for token, _ in gen:
            out.append(int(token))
            if out[-1] in engine.tokenizer.eos_token_ids:
                break
    finally:
        gen.close()
    old = next(c for c in previous['cases'] if c['label'] == label)
    report['cases'].append({'label': label, 'prompt_rendered': engine.tokenizer.decode(ids), 'token_ids': out, 'text': engine.tokenizer.decode(out), 'matches_engine': out == old['token_ids']})
engine.close()
Path('benchmarks/reports/qwen38-native-quality.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False))
