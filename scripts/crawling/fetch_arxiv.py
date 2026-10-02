"""Run collection directly, or as a module, without changing the working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arxiv_paper_recommend.corpus.crawling.arxiv import main

if __name__ == "__main__":
    main()
