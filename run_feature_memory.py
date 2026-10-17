"""Generate patents using feature memory, without baseline entry points."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.feature_memory_experiment.feature_extractor import FeatureExtractor
from experiments.feature_memory_experiment.feature_memory import FeatureMemory
from experiments.feature_memory_experiment.pipeline import FeatureMemoryPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen-plus"))
    parser.add_argument("--feature-model")
    parser.add_argument("--coverage-model")
    parser.add_argument("--mode", choices=["memory_only", "memory_tracking", "full_repair"], default="full_repair")
    parser.add_argument("--coverage-mode", choices=["llm", "hybrid", "lexical"], default="hybrid")
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.json")) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error("No input JSON files found")
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")):
        parser.error("Set OPENAI_API_KEY before generation")
    # Check every destination before incurring API charges.
    for path in paths:
        output = args.output_dir / path.name
        if output.resolve() == path.resolve() or output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor(args.feature_model or args.model, args.output_dir / "feature_cache")
    # GlobalMemory is process-wide; complete patent runs must be sequential.
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        draft = data.get("draft")
        if not isinstance(draft, (dict, str)) or not draft:
            raise ValueError(f"{path}: expected a nonempty draft object or string")
        draft_text = json.dumps(draft, ensure_ascii=False, indent=2)
        key = hashlib.sha256((draft_text + "\n" + extractor.model).encode()).hexdigest()
        features = extractor.extract(key, draft, allow_fallback=False)
        pipeline = FeatureMemoryPipeline(model=args.model, memory=FeatureMemory(features),
            mode=args.mode, coverage_mode=args.coverage_mode,
            coverage_model=args.coverage_model or args.model)
        patent = pipeline.run(draft_text)
        result = {"metadata": data.get("metadata", {}), "draft": draft, "patent": patent,
                  "feature_memory": pipeline.final_coverage,
                  "config": {"mode": args.mode, "model": args.model,
                             "feature_model": extractor.model,
                             "coverage_model": args.coverage_model or args.model,
                             "coverage_mode": args.coverage_mode}}
        output = args.output_dir / path.name
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Written: {output}")


if __name__ == "__main__":
    main()
