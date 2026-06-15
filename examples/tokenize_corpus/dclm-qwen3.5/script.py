"""Download one shard of the DCLM corpus and tokenize with the Qwen3.5 tokenizer."""

from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

from pithtrain.tasks.tokenize_corpus import TokenizeCorpusCfg, launch

if False:  # dataset already downloaded via hfd.sh
    snapshot_download(
        model_id="AI-ModelScope/dclm-baseline-1.0",
        repo_type="dataset",
        local_dir="/root/pith-train/dataset/dclm-baseline/rawtxt",
        allow_patterns="global-shard_03_of_10/local-shard_1_of_10/*.jsonl.zst",
    )

if __name__ == "__main__":
    cfg = TokenizeCorpusCfg()
    cfg.tokenizer_name = "Qwen/Qwen3.5-35B-A3B"
    cfg.source_path = Path("/root/pith-train/dataset/dclm-baseline/rawtxt/global-shard_03_of_10/local-shard_1_of_10")
    cfg.output_path = Path("/root/pith-train/dataset/dclm-baseline/toktxt/qwen3.5")
    launch(cfg)
