from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path

from voicetut_modal import VoiceTut, app

L4_GPU_USD_PER_SECOND = 0.000222
SPEAKER = "Mohamed"
NUM_STEP = 32
GUIDANCE_SCALE = 2.0
SPEED = 1.0

# One continuous request on purpose. Arabic narration is manually diacritized as fully as
# practical, while the English code-switches stay in Latin script. Paragraph breaks give
# the narrator natural transitions without resetting speaker/context between scenes.
FULL_SCRIPT = """
رَمْسِيس التَّانِي... الِاسْم دَهْ لَوَحْدُه كَانْ مَعْنَاه قُوَّة. الرَّاجِل حَكَمْ مِصْر سِتَّة وَسِتِّين سَنَة، وَبَنَى أَبُو سِمْبِل، وَكَانْ حَرْفِيًّا one of the biggest names in ancient history.

وَمِنْ أَهَمّ الحَاجَات اللِّي عَمَلْهَا إِنُّه دَخَلْ فِي صِرَاع ضَخْم مَعَ الحِيثِيِّين، وَبَعْدَهَا وِصْلُوا لِوَاحْدَة مِنْ أَقْدَم اتِّفَاقِيَّات السَّلَام المَعْرُوفَة فِي التَّارِيخ. يَعْنِي مِنْ حَرْب تِقِيلَة جِدًّا... لِـ peace treaty.

وَالأَغْرَب؟ رَمْسِيس عَاشْ لَحَدّ حَوَالِي تِسْعِين سَنَة. تِسْعِين! فِي زَمَن كَانْ مُتَوَسِّط العُمْر فِيه أَقَلّ بِكْتِير. الرَّقْم دَهْ لَوَحْدُه يِخَلِّيك تِقُول: okay... الرَّاجِل دَهْ كَانْ حَالَة اسْتِثْنَائِيَّة فِعْلًا.

نِنُطّ بَقَى لِسَنَة 2016. فَرِيق بِقِيَادَة الدُّكْتُورَة سَحَر سَلِيم، وَالدُّكْتُور زَاهِي حَوَّاس، عَمَلْ CT scan لِمُومْيَا رَمْسِيس. لِأَوَّل مَرَّة نِقْدَر نِبُصّ جُوَّه جِسْمُه بِالتَّفْصِيل... مِنْ غَيْر مَا نِلْمَس المُومْيَا أَصْلًا.

الـ CT scan بِبَسَاطَة بِيصَوِّر الجِسْم slice by slice. شَرِيحَة وَرَا شَرِيحَة، وَبَعْدَهَا الكُمْبِيُوتَر يِرَكِّبْهُم مَعْ بَعْض، فَتِقْدَر تِشُوف العَضْم وَالسِّنَان مِنْ جُوَّه، كَإِنّ الجِسْم مَفْتُوح قُدَّامَك... بَسّ مِنْ غَيْر أَيّ جِرَاحَة.

وَهِنَا ظَهَرِت المُفَاجَأَة. السِّنَان كَانِت مُتَآكِلَة بِشَكْل شَدِيد، فِيه سِنَان مَفْقُودَة، وَفِيه damage وَاضِح جِدًّا فِي الفَكّ. يَعْنِي مِشْ مُجَرَّد وَجَع سِنّ عَادِي وَخَلَاص... المَوْضُوع كَانْ أَكْبَر مِنْ كِدَه.

كَانْ فِيه خُرَّاج كَبِير عِنْد جِذْر ضِرْس فِي الفَكّ الشِّمَال. abscess يَعْنِي صَدِيد وَعَدْوَى حَوَالِين الجِذْر وَالعَضْم. وَدَهْ غَالِبًا مَعْنَاه أَلَم مُسْتَمِرّ، يَوْم وَرَا يَوْم... وَيُمْكِن لِسِنِين.

فَكِّر فِيهَا كِدَه: أَقْوَى رَاجِل فِي العَالَم وَقْتَهَا، عِنْدُه إِمْبِرَاطُورِيَّة، دَهَب، أَطِبَّاء وَكَهَنَة... وَمَعَ كُلّ دَهْ بُقُّه بِيُوجَعُه، وَمَحَدِّش قَادِر يِخَلِّصُه مِنَ المُشْكِلَة. That's the plot twist.

وَالأَغْرَب إِنّ نَفْس المُشْكِلَة تَقْرِيبًا كَانِت بْتِحْصَل لِأَفْقَر فَلَّاح فِي الدِّلْتَا. مَلِك عَلَى العَرْش وَفَلَّاح بَسِيط... different lives, same teeth problem. وَالسُّؤَال الحَقِيقِي هِنَا: لِيه؟

عَشَان نِجَاوِب، لَازِم نِبْعَد شُوَيَّة عَنِ المَقَابِر وَالدَّهَب، وَنِسْأَل عَنْ daily life. المِصْرِي القَدِيم كَانْ بِيِفْطَر إِيه؟ بِيِشْتَغَل إِزَّاي؟ بِيِتْخَانِق مَعَ مُدِيرُه عَلَى إِيه؟ وَحَتَّى أَقْدَم بَرْدِيَّة عِنْدِنَا... طِلْعِت فِي الآخِر شِبْه schedule مُوَظَّف عَادِي. وَدِي بِدَايَة الحِكَايَة.
""".strip()


def wav_duration_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@app.local_entrypoint()
def benchmark(output_dir: str = "voicetut-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Deliberately do NOT call speakers.remote() first. This must be the first remote call
    # so elapsed time includes a real cold container/model start plus the full synthesis.
    tts = VoiceTut()
    started = time.perf_counter()
    audio = tts.synthesize.remote(
        FULL_SCRIPT,
        SPEAKER,
        NUM_STEP,
        GUIDANCE_SCALE,
        SPEED,
        True,
    )
    request_seconds = time.perf_counter() - started
    audio_seconds = wav_duration_seconds(audio)
    rtf = request_seconds / audio_seconds if audio_seconds else None

    wav_path = target / "voicetut-ramses-one-request-full-diacritics.wav"
    wav_path.write_bytes(audio)
    (target / "input-script.txt").write_text(FULL_SCRIPT, encoding="utf-8")

    result = {
        "experiment": "voicetut-ramses-one-request-full-diacritics",
        "model": "mohammedaly22/VoiceTut-TTS",
        "gpu": "L4",
        "speaker": SPEAKER,
        "num_step": NUM_STEP,
        "guidance_scale": GUIDANCE_SCALE,
        "speed": SPEED,
        "request_count": 1,
        "scene_paragraphs": 10,
        "input_characters": len(FULL_SCRIPT),
        "request_seconds_including_cold_start": request_seconds,
        "audio_seconds": audio_seconds,
        "real_time_factor_including_cold_start": rtf,
        "bytes": len(audio),
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_request_elapsed": request_seconds * L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_with_30s_scaledown": (request_seconds + 30.0) * L4_GPU_USD_PER_SECOND,
        "note": "One synthesize.remote call contains all ten Ramses paragraphs. Arabic words are manually diacritized; English code-switches remain in Latin script. No speaker warm-up call is made before timing.",
    }
    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"VOICETUT_ONE_REQUEST request_seconds={request_seconds:.3f} "
        f"audio_seconds={audio_seconds:.3f} rtf={rtf:.3f} bytes={len(audio)} "
        f"characters={len(FULL_SCRIPT)}"
    )
    print("VOICETUT_BENCHMARK_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
