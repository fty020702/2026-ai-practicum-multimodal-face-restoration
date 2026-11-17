from __future__ import annotations

"""DEPRECATED compatibility launcher -- kept so old links still work.

The Gradio skeleton was REMOVED on 2026-11-17: installing gradio 4.x into the
shared ChatAI env upgrades pydantic / rich / typer and breaks the pre-existing
fschat / spacy / streamlit / weasel packages. The course demo is now
`app/demo_server.py`, a ZERO-NEW-DEPENDENCY stdlib HTTP server that drives the
real BF16 FLUX.2 [klein] 4B pipeline and includes a robustness / boundary panel.

    D:/Anaconda/envs/ChatAI/python.exe app/demo_server.py --port 7860
    D:/Anaconda/envs/ChatAI/python.exe app/gradio_demo.py   # same thing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_server import main  # noqa: E402

if __name__ == "__main__":
    print("[demo] app/gradio_demo.py is deprecated; forwarding to "
          "app/demo_server.py (no gradio required)", flush=True)
    sys.exit(main())
