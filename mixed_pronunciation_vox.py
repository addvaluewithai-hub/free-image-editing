from __future__ import annotations

from mixed_pronunciation_common import run_benchmark
from vox_oneshot_modal import VoxEnglishOneShot, app

VOICE_DESCRIPTION = (
    "Natural bilingual Arabic and American-English educational narrator. Calm teacher-like delivery, clear diction, "
    "smooth code-switching, and one consistent voice. Do not intentionally read XML tag names aloud."
)


@app.local_entrypoint()
def benchmark(output_dir: str = "mixed-pronunciation-vox") -> None:
    tts = VoxEnglishOneShot()

    def synth(text: str, _index: int) -> bytes:
        return tts.synthesize.remote(text, VOICE_DESCRIPTION, 2.0, 10)

    def control(text: str) -> bytes:
        return tts.synthesize.remote(text, VOICE_DESCRIPTION, 2.0, 10)

    run_benchmark(
        output_dir=output_dir,
        model_name="openbmb/VoxCPM2",
        synth_chunk=synth,
        synth_control=control,
        notes=(
            "VoxCPM2 Voice Design with optimize=True, cfg=2.0, inference_timesteps=10. "
            "The exact SSML-looking text reaches VoxCPM2; there is no custom SSML parser or IPA preprocessing."
        ),
    )
