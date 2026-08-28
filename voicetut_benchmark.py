from __future__ import annotations

import io
import json
import statistics
import time
import wave
from pathlib import Path

from voicetut_modal import VoiceTut, app

L4_GPU_USD_PER_SECOND = 0.000222
SPEAKER = "Mohamed"
NUM_STEP = 32
GUIDANCE_SCALE = 2.0
SPEED = 1.0

SCENES = [
    ("01-ramses-intro", "رمسيس التاني... الاسم ده لوحده كان معناه قوة. الراجل حكم مصر ستة وستين سنة، وبنى أبو سمبل، وكان حرفيًا one of the biggest names in ancient history."),
    ("02-peace-treaty", "ومن أهم الحاجات اللي عملها إنه دخل في صراع ضخم مع الحيثيين، وبعدها وصلوا لواحدة من أقدم اتفاقيات السلام المعروفة في التاريخ. يعني من حرب تقيلة جدًا... لـ peace treaty."),
    ("03-long-life", "والأغرب؟ رمسيس عاش لحد حوالي تسعين سنة. تسعين! في زمن كان متوسط العمر فيه أقل بكتير. الرقم ده لوحده يخليك تقول: okay... الراجل ده كان حالة استثنائية فعلًا."),
    ("04-ct-scan", "نقفز بقى لسنة 2016. فريق بقيادة الدكتورة سحر سليم والدكتور زاهي حواس عمل CT scan لمومياء رمسيس. لأول مرة نقدر نبص جوه جسمه بالتفصيل... من غير ما نلمس المومياء أصلًا."),
    ("05-ct-explainer", "الـ CT scan ببساطة بيصور الجسم slice by slice. شريحة ورا شريحة، وبعدها الكمبيوتر يركبهم مع بعض، فتقدر تشوف العضم والأسنان من جوه كإن الجسم مفتوح قدامك... بس من غير أي جراحة."),
    ("06-dental-damage", "وهنا ظهرت المفاجأة. الأسنان كانت متآكلة بشكل شديد، في أسنان مفقودة، وفي damage واضح جدًا في الفك. يعني مش مجرد وجع سن عادي وخلاص... الموضوع كان أكبر من كده."),
    ("07-abscess", "كان فيه خراج كبير عند جذر ضرس في الفك الشمال. abscess يعني صديد وعدوى حوالين الجذر والعظم. وده غالبًا معناه ألم مستمر، يوم ورا يوم... ويمكن لسنين."),
    ("08-powerful-king-in-pain", "فكر فيها كده: أقوى راجل في العالم وقتها، عنده إمبراطورية، دهب، أطباء وكهنة... ومع كل ده بُقه بيوجعه ومحدش قادر يخلّصه من المشكلة. That's the plot twist."),
    ("09-king-vs-farmer", "والأغرب إن نفس المشكلة تقريبًا كانت بتحصل لأفقر فلاح في الدلتا. ملك على العرش وفلاح بسيط... different lives, same teeth problem. والسؤال الحقيقي هنا: ليه؟"),
    ("10-daily-life", "عشان نجاوب، لازم نبعد شوية عن المقابر والدهب ونسأل عن daily life. المصري القديم كان بيفطر إيه؟ بيشتغل إزاي؟ بيتخانق مع مديره على إيه؟ وحتى أقدم بردية عندنا... طلعت في الآخر شبه schedule موظف عادي. ودي بداية الحكاية."),
]


def wav_duration_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@app.local_entrypoint()
def benchmark(output_dir: str = "voicetut-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    tts = VoiceTut()
    speakers = tts.speakers.remote()
    print(f"VOICETUT_SPEAKERS={speakers}")
    if SPEAKER not in speakers:
        raise RuntimeError(f"Expected built-in speaker {SPEAKER!r} not found. Available: {speakers}")

    records = []
    generation_times = []
    audio_durations = []
    overall_start = time.perf_counter()

    for index, (slug, text) in enumerate(SCENES, start=1):
        started = time.perf_counter()
        audio = tts.synthesize.remote(
            text,
            SPEAKER,
            NUM_STEP,
            GUIDANCE_SCALE,
            SPEED,
            True,
        )
        generation_seconds = time.perf_counter() - started
        audio_seconds = wav_duration_seconds(audio)
        rtf = generation_seconds / audio_seconds if audio_seconds else None
        generation_times.append(generation_seconds)
        audio_durations.append(audio_seconds)

        path = target / f"{index:02d}-{slug}.wav"
        path.write_bytes(audio)
        records.append(
            {
                "index": index,
                "slug": slug,
                "speaker": SPEAKER,
                "text": text,
                "generation_seconds": generation_seconds,
                "audio_seconds": audio_seconds,
                "real_time_factor": rtf,
                "bytes": len(audio),
            }
        )
        print(
            f"VOICETUT_SCENE index={index} slug={slug} speaker={SPEAKER} "
            f"generation_seconds={generation_seconds:.3f} audio_seconds={audio_seconds:.3f} "
            f"rtf={rtf:.3f} bytes={len(audio)}"
        )

    total_wall = time.perf_counter() - overall_start
    warm = records[1:]
    warm_generation = [r["generation_seconds"] for r in warm]
    warm_audio = [r["audio_seconds"] for r in warm]
    warm_total_gen = sum(warm_generation)
    warm_total_audio = sum(warm_audio)
    warm_rtf = warm_total_gen / warm_total_audio if warm_total_audio else None
    gpu_raw = total_wall * L4_GPU_USD_PER_SECOND
    gpu_with_tail = (total_wall + 30.0) * L4_GPU_USD_PER_SECOND
    warm_cost_per_audio_min = (
        warm_rtf * 60.0 * L4_GPU_USD_PER_SECOND if warm_rtf is not None else None
    )

    result = {
        "experiment": "voicetut-ramses-10-scene-fixed-speaker",
        "model": "mohammedaly22/VoiceTut-TTS",
        "gpu": "L4",
        "speaker": SPEAKER,
        "num_step": NUM_STEP,
        "guidance_scale": GUIDANCE_SCALE,
        "speed": SPEED,
        "scene_count": len(SCENES),
        "first_call_seconds": generation_times[0],
        "warm_average_generation_seconds": statistics.mean(warm_generation),
        "warm_min_generation_seconds": min(warm_generation),
        "warm_max_generation_seconds": max(warm_generation),
        "total_generation_wall_seconds": total_wall,
        "total_audio_seconds": sum(audio_durations),
        "warm_real_time_factor": warm_rtf,
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_raw_wall": gpu_raw,
        "estimated_gpu_usd_with_30s_scaledown": gpu_with_tail,
        "estimated_warm_gpu_usd_per_audio_minute": warm_cost_per_audio_min,
        "estimated_warm_audio_minutes_per_30_usd": (
            30.0 / warm_cost_per_audio_min if warm_cost_per_audio_min else None
        ),
        "available_speakers": speakers,
        "scenes": records,
        "note": "All ten clips use the exact same built-in speaker (Mohamed) to test speaker consistency, Egyptian Arabic, English code-switching, cold start and warm RTF. GPU estimates use the L4 rate only.",
    }
    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("VOICETUT_BENCHMARK_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
