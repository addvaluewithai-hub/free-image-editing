from __future__ import annotations

from mixed_pronunciation_common import run_benchmark
from chatterbox_turbo_modal import ChatterboxTurbo, app


@app.local_entrypoint()
def benchmark(output_dir: str = "mixed-pronunciation-chatterbox") -> None:
    tts = ChatterboxTurbo()

    def synth(text: str, index: int) -> bytes:
        return tts.synthesize.remote(text, 4100 + index, 0.8, 0.95, 1000, 1.2)

    def control(text: str) -> bytes:
        return tts.synthesize.remote(text, 4199, 0.8, 0.95, 1000, 1.2)

    run_benchmark(
        output_dir=output_dir,
        model_name="ResembleAI/chatterbox-turbo",
        synth_chunk=synth,
        synth_control=control,
        notes=(
            "Chatterbox Turbo is an English-focused model; this intentionally stress-tests unsupported mixed Arabic/English input. "
            "The exact SSML-looking text reaches the model with no SSML parser or IPA preprocessing."
        ),
    )
