SCRIPT = """At 2:17 in the morning, the Northstar Deep Space Array went silent. No alarms. No explosion. Just twelve enormous antennas, frozen under a moonless sky, suddenly pointing at nothing.

For engineer Elena Marquez, that silence was worse than noise. Three minutes earlier, Northstar had been tracking a probe two hundred and eighty million miles from Earth. Its signal was weak, barely more than a whisper in the static, but it was there. Then—gone. Elena stared at the telemetry and said the one thing nobody in the control room wanted to hear: “That wasn't the probe.”

At first, the team blamed the weather. Easy answer. Comfortable answer. Also... completely wrong. Wind speed was normal. Power was stable. The cryogenic receivers were cold, the clocks were synchronized, and every diagnostic light was green. Perfect. Which is exactly when you start to worry.

Then a junior technician noticed something strange. Antenna seven had moved by zero point three degrees. That sounds tiny. It isn't. At that distance, zero point three degrees is the difference between listening to a spacecraft... and listening to empty space.

Elena leaned toward the console. “Run the command history again.” This time her voice was lower. Slower. Nobody joked. Nobody moved. The log showed a steering command at 2:16:42 a.m. It had valid credentials. It had passed every security check. And according to the system, Elena herself had sent it.

She hadn't.

Now the room changed. Curiosity became tension. Tension became fear. Someone killed the external network link. Another engineer started reading the command stack line by line. And then, buried between two routine calibration messages, they found it: a malformed packet with a timestamp from tomorrow.

Tomorrow.

For about five seconds, the entire room just stared at the screen. Then Marcus, the systems lead, broke the silence: “Great. So either we've been hacked by a time traveler... or our clock is lying to us.” A few people laughed. Not because it was funny. Because sometimes your brain needs somewhere to put the panic.

The clock wasn't lying. But one backup server was. A firmware bug had pushed its date forward by exactly twenty-four hours, causing it to replay an old steering command as if it were new. No attacker. No sabotage. No science-fiction mystery. Just one tiny software error, hiding inside a machine that had worked perfectly for six years.

And here is the part that still bothers Elena. The probe's signal returned at 2:29 a.m.—twelve minutes after it vanished. Twelve minutes doesn't sound like much. But when your spacecraft is millions of miles away, twelve minutes feels enormous. You cannot walk outside and fix it. You cannot restart the universe. You can only send a command into the dark... and wait.

At 2:31, the first clean telemetry packet arrived. Battery voltage: normal. Guidance: normal. Memory: intact. The room erupted. One technician actually hugged a printer. Marcus claimed he had never been worried, which was a spectacular lie.

By sunrise, Northstar was tracking the probe again, the faulty server was isolated, and somebody had written “NO TIME TRAVEL” on the whiteboard in red marker.

But the lesson wasn't really about a bad clock. It was about confidence. Complex systems rarely fail with dramatic sparks and smoke. Sometimes they fail quietly, politely, with every status light glowing green. And the most dangerous sentence in any control room might be the simplest one: “It can't be that.”

Because sometimes... it absolutely can."""

QWEN_INSTRUCT = """Premium cinematic English documentary narration with one consistent male narrator identity throughout. Let the delivery evolve naturally with the story: confident and mysterious opening; quiet curiosity; restrained suspense; lower and slower delivery during the command-history reveal; genuine shock on 'Tomorrow'; dry understated humor on the time-traveler line; controlled urgency and fear; thoughtful empathy during the waiting section; strong relief when telemetry returns; playful warmth on the printer joke; then a calm, reflective, memorable ending. Use natural pauses, dynamic pacing, and believable emotion. Never become cartoonish or over-theatrical."""

VOX_VOICE_DESCRIPTION = """Premium English male documentary narrator, early thirties, warm resonant voice, highly natural and believable, clear American English diction, consistent voice identity, cinematic but not theatrical. Dynamically follow the emotional arc of the story: mysterious confidence, curiosity, restrained suspense, lower slower tension, genuine surprise, dry understated humor, controlled fear and urgency, thoughtful empathy, strong relief, playful warmth, then a calm reflective ending. Use natural pauses and varied pacing while keeping the same narrator throughout."""
