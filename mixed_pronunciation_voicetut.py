from __future__ import annotations

from mixed_pronunciation_common import run_benchmark
from voicetut_modal import VoiceTut, app


@app.local_entrypoint()
def benchmark(output_dir: str = "mixed-pronunciation-voicetut") -> None:
    tts = VoiceTut()

    def synth(text: str, _index: int) -> bytes:
        return tts.synthesize.remote(text, "Mohamed", 32, 2.0, 1.0, True)

    def control(text: str) -> bytes:
        return tts.synthesize.remote(text, "Mohamed", 32, 2.0, 1.0, True)

    run_benchmark(
        output_dir=output_dir,
        model_name="mohammedaly22/VoiceTut-TTS",
        synth_chunk=synth,
        synth_control=control,
        notes=(
            "VoiceTut speaker Mohamed, language=arz, normalize=True, num_step=32. "
            "This model is included because the requested use case is Arabic-English code-switching. "
            "No custom SSML parser or IPA preprocessing is applied before synthesis. Research comparison only; commercial licensing remains ambiguous due to the OmniVoice base-weight licensing chain."
        ),
    )
