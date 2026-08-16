# Design notes

Long-form reasoning behind two subsystems that are summarized in the
[README](README.md). Read this if you want to know *why* they are shaped the way
they are; the README is enough to run them.

---

# Risk tiering

Isolated alerts miss patterns. One person on one camera is routine — but the
same few minutes producing *loitering out front, a person on the driveway, and
a person at the back door* is someone moving around the property. Risk tiering
scores that.

```
score(now) = Σ over recent events:  weight(label) × camera_mult × night_mult
                                    × repeat_dampening^rank
                                    × 0.5 ^ (age / half_life)
```

**Every event adds points; silence decays them.** Half-life is ~10 minutes, so
blips fade and patterns escalate. Score ranges map to tiers — `QUIET`,
`NOTICE`, `ELEVATED`, `URGENT`.

**The score is derived, never stored.** A decayed sum over the event log is
mathematically identical to keeping a running level, but has no mutable state to
drift or corrupt — and it makes history replayable. The `risk_*` columns written
to `events.db` are an audit trail, never read back for a decision. This is the
same pattern as the cooldown, which is derived from the most recent alerted row
rather than a side file.

**The LLM never produces the score.** Asked for a number, it would produce a
plausible, non-reproducible one. The model contributes a label; deterministic
Python does the arithmetic.

**Repeat dampening** was found during calibration — see the war story
[*A raw event sum scores frame count, not activity*](WAR-STORIES.md#a-raw-event-sum-scores-frame-count-not-activity).
The newest event of each camera+label counts in full; each older repeat counts
half again. Distinct signals stack; re-observing one open garage door does not.

**Time of day belongs to the score, not the label space.** The night multiplier
(×3) is applied from the event's timestamp, so a `*_NIGHT` label would
double-count the same signal. In 28 days the model never once emitted
`AT_NIGHT_PERSON`, so the escalation it existed to trigger never fired at all.
Those labels were retired and a test now keeps them from creeping back. A
log-only label with a low base weight still escalates correctly at 3 a.m. — a
daytime mislabel stays `QUIET`, a real 2 a.m. hit reaches `ELEVATED` in one frame.

All knobs live in `cameras.yaml`: a global `_risk` block (half-life, window,
dampening, night hours, tier thresholds) plus a per-camera `risk` block
(multiplier, label weights). No `_risk` block means the feature is off.

## Two tuning tools, two different jobs

`replay.py` re-scores labels **already in the log**, so it tunes *weights*
without re-running the vision model — two weeks of history in seconds.

It is structurally blind to a **prompt** change, which alters which labels get
produced in the first place. That needs the original frames and a second pass
through the model, which is `ab_prompt.py`. Conflating the two is why a prompt
regression can sit unmeasured for weeks: the weight tuner will happily report
that nothing changed.

## The prompt tuner needs a noise floor, and it is not small

`Modelfile` sets `temperature 0.1`, not 0. So a single classification is a
*sample* of what a prompt answers, not the answer — and an A/B that runs each
config once is comparing two samples.

Measured on the driveway camera (2026-08-09): the **live config, re-run on the
same 59 images, agreed with itself on 44 of them — 75%.** Two candidate prompts
had been rejected that day on disagreement rates of 32% and 49%, both inside
that floor. Neither was really measured.

The spread is not uniform, and that detail is what saves the method. Pooling
four draws of the live config:

| Label family | Unanimous across 4 draws |
|---|---|
| `GATE_*` / `NONE` | 26/50 (52%) |
| `PERSON` | 8/8 (100%) |
| `KNOWN_VEHICLE` | 1/1 (100%) |

**All of the instability is in one label family** — a distant chain-link gate,
often backlit or seen at an angle, is a genuinely borderline call. Occupancy
labels never wavered. So the tool measures occupancy fine; gate-state
disagreements simply are not findings on that camera.

`ab_prompt.py --repeat N` draws each config N times per frame and reports
self-consistency per label, then states outright whether the A-vs-B difference
clears the floor. Run it before believing any prompt verdict — and per camera,
since the floor is a property of the view, not of the model.

The general lesson is not about this project: **any eval that samples a
stochastic model once per item is measuring the sampler as much as the change.
Establish the floor first, and check whether it is concentrated somewhere.**

---

# Watchdog: knowing the pipeline died

A system whose normal output is silence cannot tell you it stopped. Two
production outages — 7 hours and 40% of a day, both an unreachable Ollama —
passed unnoticed. `watchdog.py` runs from Task Scheduler every 5 minutes and
checks four things (staleness, error ratio, Ollama reachability, disk). Two of
those checks are non-obvious.

## "No events" would have caught neither outage

BlueIris kept firing and the script kept logging. On 2026-07-25 that meant
**931 rows, every one an `error: ollama call failed`**, including an 11-hour
stretch from 04:00 to 14:00 in which not a single frame was classified.

A silence check sees a perfectly healthy event rate and stays quiet. The
**error-ratio** check is what catches a dead model; silence only catches a dead
NVR or a dead box. Both checks are needed, and the obvious one is the weaker.

## A fixed silence threshold cannot work

The p99 gap between events — the quantity a silence check is really testing —
swings **~30× across the day**: 137 seconds at 4pm against 70 minutes at 5am.
"20 minutes of quiet" is meaningless at night and alarming at midday.

So the threshold is **learned per hour-of-day from the event log itself**,
floored at 45 minutes by day and 180 minutes inside the night window, where
"quiet house" and "dead pipeline" genuinely look alike. Gaps longer than the
ceiling are excluded from the learning set, so an outage cannot teach the
watchdog to sleep through the next one.

## Notify on state change, not per run

A check that fires every 5 minutes while a condition persists is a check people
turn off. The watchdog emails on a *confirmed state change*. Replayed against 28
days of history: **20 emails covering 4 real incidents**, instead of one every
5 minutes.

It is strictly a sidecar — it opens `events.db` read-only (`mode=ro`) and never
sits in the alert path, so a bug in the watchdog cannot take down alerting.

---

# Latency and the keep-alive knob

Loading the ~8 GB model into the GPU takes about 7 seconds, so in theory an
alert that arrives at a cold model pays that reload on top of inference.
`OLLAMA_KEEP_ALIVE` tells Ollama how long to keep the model resident after a
request, the intent being that sparse motion alerts don't each trigger a reload.
`30m` is the default here; `-1` keeps it warm forever, which holds the VRAM
permanently and blocks other large models from loading alongside it. The box's
own idle default may be only seconds, and this per-request value overrides it.

**In practice, tuning it did not meaningfully change per-alert latency in this
setup.** End-to-end time was acceptable either way, so the root cause was never
chased down. Treat it as a reasonable knob to try, not a proven fix — recorded
here rather than quietly dropped, because "I tried this and it didn't move the
needle" is worth as much to the next person as a fix would be.

What *does* dominate latency is GPU residency: `ollama ps` should report
`100% GPU`. CPU offload means the model doesn't fit in VRAM, and that costs far
more than any reload.
