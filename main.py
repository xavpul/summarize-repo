#!/usr/bin/env python3
import sys
import glob
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains.summarize.chain import load_summarize_chain


def main():
    # 1. Prompt for repo root
    repo = (
        Path(input("Absolute or relative path to the repo root: ").strip())
        .expanduser()
        .resolve()
    )
    if not repo.is_dir():
        print("❌  That path is not a directory. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 2. Collect files
    exts = ("*.py", "*.md", "*.txt", "*.ini", "*.toml")
    files = []
    for ext in exts:
        files.extend(repo.rglob(ext))
    if not files:
        print("❌  No matching files found. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"📁  Found {len(files)} files to summarize...")

    # 3. Load & split
    docs = []
    for f in files:
        try:
            docs.extend(TextLoader(str(f), encoding="utf-8").load())
        except Exception as e:
            print(f"⚠️  Skipping {f}: {e}", file=sys.stderr)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # smaller chunk to fit in 8 GB
        chunk_overlap=100,
    )
    splits = splitter.split_documents(docs)

    # 4. Initialize LLM & chain
    try:
        llm = OllamaLLM(model="codellama:7b")
    except Exception as e:
        print("❌  Failed to initialize Ollama LLM:", e, file=sys.stderr)
        sys.exit(1)
    try:
        chain = load_summarize_chain(llm, chain_type="map_reduce")
    except Exception as e:
        print("❌  Failed to load summarize chain:", e, file=sys.stderr)
        sys.exit(1)

    # 5. Run summarization
    try:
        result = chain.invoke({"input_documents": splits})
        # extract text from the returned dict
        summary = result.get("output_text") or result.get("summary") or str(result)
    except Exception as e:
        print("❌  Summarization failed:", e, file=sys.stderr)
        sys.exit(1)

    # 6. Save to summary.txt in CWD
    out_path = Path.cwd() / "summary.txt"
    try:
        out_path.write_text(summary, encoding="utf-8")
    except Exception as e:
        print("❌  Failed to write summary:", e, file=sys.stderr)
        sys.exit(1)

    print(f"✅  Summary saved to {out_path}")


if __name__ == "__main__":
    main()
