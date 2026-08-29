from __future__ import annotations

from mixed_pronunciation_common import run_benchmark
from qwen17_tts_modal import Qwen17TTS, app

INSTRUCT = (
    "Natural bilingual educational narration. Speak the Arabic naturally and switch cleanly to American English "
    "for embedded English. Do not intentionally read XML tag names aloud. If phoneme pronunciation hints are "
    "supported by the model, follow them. Keep a calm teacher-like delivery."
)


@app.local_entrypoint()
def benchmark(output_dir: str = "mixed-pronunciation-qwen17") -> None:
    tts = Qwen17TTS()

    def synth(text: str, _index: int) -> bytes:
        return tts.synthesize.remote(text, "Ryan", INSTRUCT, "Auto")

    def control(text: str) -> bytes:
        return tts.synthesize.remote(text, "Ryan", INSTRUCT, "English")

    run_benchmark(
        output_dir=output_dir,
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        synth_chunk=synth,
        synth_control=control,
        notes=(
            "Ryan built-in voice, FlashAttention 2, bfloat16. Main chunks use language=Auto and official 1.7B instruct control. "
            "No preprocessing strips or interprets the SSML/IPA tags before model inference."
        ),
    )
