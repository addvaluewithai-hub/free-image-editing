from __future__ import annotations

from io import BytesIO
import json
import statistics
import time
import wave
from pathlib import Path

from modal_app import VoxTTS, app

L4_GPU_USD_PER_SECOND = 0.000222
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 10

BASE_VOICE = (
    "Egyptian male narrator in his early thirties, natural Cairo accent, warm documentary storyteller, "
    "clear diction, expressive but believable, medium pace, consistent voice identity"
)

SCENES = [
    (
        "01-ramses-intro",
        "confident and energetic, slight sense of awe, strong opening emphasis",
        "رمسيس التاني... الاسم ده لوحده كان معناه قوة. الراجل حكم مصر ستة وستين سنة، وبنى أبو سمبل، وكان حرفيًا one of the biggest names in ancient history.",
    ),
    (
        "02-peace-treaty",
        "authoritative documentary tone, measured pace, proud historical emphasis",
        "ومن أهم الحاجات اللي عملها إنه دخل في صراع ضخم مع الحيثيين، وبعدها وصلوا لواحدة من أقدم اتفاقيات السلام المعروفة في التاريخ. يعني من حرب تقيلة جدًا... لـ peace treaty.",
    ),
    (
        "03-long-life",
        "surprised and impressed, slightly slower on the key number",
        "والأغرب؟ رمسيس عاش لحد حوالي تسعين سنة. تسعين! في زمن كان متوسط العمر فيه أقل بكتير. الرقم ده لوحده يخليك تقول: okay... الراجل ده كان حالة استثنائية فعلًا.",
    ),
    (
        "04-ct-scan",
        "curious investigative tone, modern science documentary feel",
        "نقفز بقى لسنة 2016. فريق بقيادة الدكتورة سحر سليم والدكتور زاهي حواس عمل CT scan لمومياء رمسيس. لأول مرة نقدر نبص جوه جسمه بالتفصيل... من غير ما نلمس المومياء أصلًا.",
    ),
    (
        "05-ct-explainer",
        "clear teacher-like explanation, friendly and precise, slightly slower pace",
        "الـ CT scan ببساطة بيصور الجسم slice by slice. شريحة ورا شريحة، وبعدها الكمبيوتر يركبهم مع بعض، فتقدر تشوف العضم والأسنان من جوه كإن الجسم مفتوح قدامك... بس من غير أي جراحة.",
    ),
    (
        "06-dental-damage",
        "serious and concerned, controlled dramatic pause before the reveal",
        "وهنا ظهرت المفاجأة. الأسنان كانت متآكلة بشكل شديد، في أسنان مفقودة، وفي damage واضح جدًا في الفك. يعني مش مجرد وجع سن عادي وخلاص... الموضوع كان أكبر من كده.",
    ),
    (
        "07-abscess",
        "tense medical-storytelling tone, empathetic, emphasize the painful detail",
        "كان فيه خراج كبير عند جذر ضرس في الفك الشمال. abscess يعني صديد وعدوى حوالين الجذر والعظم. وده غالبًا معناه ألم مستمر، يوم ورا يوم... ويمكن لسنين.",
    ),
    (
        "08-powerful-king-in-pain",
        "ironic dramatic storytelling, slightly incredulous, emotional contrast",
        "فكر فيها كده: أقوى راجل في العالم وقتها، عنده إمبراطورية، دهب، أطباء وكهنة... ومع كل ده بُقه بيوجعه ومحدش قادر يخلّصه من المشكلة. That's the plot twist.",
    ),
    (
        "09-king-vs-farmer",
        "conversational and reflective, make the comparison feel surprising but natural",
        "والأغرب إن نفس المشكلة تقريبًا كانت بتحصل لأفقر فلاح في الدلتا. ملك على العرش وفلاح بسيط... different lives, same teeth problem. والسؤال الحقيقي هنا: ليه؟",
    ),
    (
        "10-daily-life",
        "playful curious ending, inviting tone, build anticipation for the next section",
        "عشان نجاوب، لازم نبعد شوية عن المقابر والدهب ونسأل عن daily life. المصري القديم كان بيفطر إيه؟ بيشتغل إزاي؟ بيتخانق مع مديره على إيه؟ وحتى أقدم بردية عندنا... طلعت في الآخر شبه schedule موظف عادي. ودي بداية الحكاية.",
    ),
]


def wav_duration_seconds(data: bytes) -> float:
    with wave.open(BytesIO(data), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


@app.local_entrypoint()
def benchmark(output_dir: str = "vox-benchmark-output") -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    tts = VoxTTS()
    rows = []
    call_durations: list[float] = []
    audio_durations: list[float] = []
    wall_start = time.perf_counter()

    for index, (slug, expression, text) in enumerate(SCENES, start=1):
        voice_description = BASE_VOICE + ", " + expression
        started = time.perf_counter()
        wav_bytes = tts.synthesize.remote(
            text,
            voice_description,
            CFG_VALUE,
            INFERENCE_TIMESTEPS,
            True,
        )
        elapsed = time.perf_counter() - started
        audio_seconds = wav_duration_seconds(wav_bytes)
        rtf = elapsed / audio_seconds if audio_seconds else None

        path = target / f"{index:02d}-{slug}.wav"
        path.write_bytes(wav_bytes)

        row = {
            "index": index,
            "slug": slug,
            "expression": expression,
            "text": text,
            "generation_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "real_time_factor": rtf,
            "bytes": len(wav_bytes),
        }
        rows.append(row)
        call_durations.append(elapsed)
        audio_durations.append(audio_seconds)
        print(
            f"VOX_SCENE index={index} slug={slug} generation_seconds={elapsed:.3f} "
            f"audio_seconds={audio_seconds:.3f} rtf={rtf:.3f} bytes={len(wav_bytes)}"
        )

    total_wall = time.perf_counter() - wall_start
    warm_calls = call_durations[1:]
    warm_audio = audio_durations[1:]
    warm_call_total = sum(warm_calls)
    warm_audio_total = sum(warm_audio)

    raw_gpu_cost = total_wall * L4_GPU_USD_PER_SECOND
    conservative_gpu_cost = (total_wall + 30.0) * L4_GPU_USD_PER_SECOND
    warm_cost = warm_call_total * L4_GPU_USD_PER_SECOND
    warm_audio_minutes = warm_audio_total / 60.0
    warm_cost_per_audio_minute = warm_cost / warm_audio_minutes if warm_audio_minutes else None

    result = {
        "experiment": "voxcpm2-ramses-10-scene-egyptian-mixed-english-expressions",
        "model": "openbmb/VoxCPM2",
        "gpu": "L4",
        "cfg_value": CFG_VALUE,
        "inference_timesteps": INFERENCE_TIMESTEPS,
        "scene_count": len(SCENES),
        "first_call_seconds": call_durations[0],
        "warm_average_generation_seconds": statistics.mean(warm_calls),
        "warm_min_generation_seconds": min(warm_calls),
        "warm_max_generation_seconds": max(warm_calls),
        "total_generation_wall_seconds": total_wall,
        "total_audio_seconds": sum(audio_durations),
        "overall_real_time_factor": total_wall / sum(audio_durations),
        "warm_real_time_factor": warm_call_total / warm_audio_total,
        "l4_gpu_usd_per_second": L4_GPU_USD_PER_SECOND,
        "estimated_gpu_usd_raw_wall": raw_gpu_cost,
        "estimated_gpu_usd_with_30s_scaledown": conservative_gpu_cost,
        "estimated_warm_gpu_usd_per_audio_minute": warm_cost_per_audio_minute,
        "estimated_warm_audio_minutes_per_30_usd": (
            30.0 / warm_cost_per_audio_minute if warm_cost_per_audio_minute else None
        ),
        "scenes": rows,
        "note": (
            "All ten clips are sequential calls to the same VoxTTS handle. The first call includes cold-start/compile overhead; "
            "calls 2-10 estimate warm performance. GPU cost is estimated from L4 time only and excludes CPU/RAM/network."
        ),
    }

    (target / "benchmark.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("VOX_BENCHMARK_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
