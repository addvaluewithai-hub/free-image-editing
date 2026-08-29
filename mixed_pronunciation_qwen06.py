from __future__ import annotations

from mixed_pronunciation_common import run_benchmark
from qwen3_tts_modal import Qwen3TTS, app


@app.local_entrypoint()
def benchmark(output_dir: str = "mixed-pronunciation-qwen06") -> None:
    tts = Qwen3TTS()

    def synth(text: str, _index: int) -> bytes:
        return tts.synthesize.remote(text, "Ryan", None, "Auto")

    def control(text: str) -> bytes:
        return tts.synthesize.remote(text, "Ryan", None, "English")

    run_benchmark(
        output_dir=output_dir,
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        synth_chunk=synth,
        synth_control=control,
        notes=(
            "Ryan built-in voice. Main mixed-language chunks use language=Auto. "
            "The official 0.6B CustomVoice implementation ignores instruct, so no instruction is supplied. "
            "This is a raw SSML-looking input test; no tags are stripped before inference."
        ),
    )
