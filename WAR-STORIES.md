# War stories

The non-obvious problems that came with wiring an LLM into a Windows NVR — most
hit in production, one headed off by design. The kind of thing that doesn't show
up in a unit test, and the reason [`events.db`](README.md#inspecting-the-log)
logs every frame instead of only the interesting ones.

Each entry ends with the generalized lesson, which is the part that transfers.

---

## `&ALERT_PATH` sometimes arrives as a bare filename

Alerts were logged as "image not found" even though BlueIris clearly passed
*something*. Depending on BlueIris version and settings, the `&ALERT_PATH` macro
expands to just a filename with no folder.

Fix: `resolve_image()` tries the path as given, then falls back to
`ALERT_IMAGE_DIR / filename`.

> **Lesson:** don't trust an upstream integration's string format — degrade
> gracefully instead of assuming an absolute path.

## Service accounts don't share your login's secret store

*(Headed off by design rather than gotten burned by.)*

The Gmail App Password lives in **Windows Credential Manager, which is per-user
and DPAPI-encrypted**. BlueIris often runs under a *different* account (e.g.
`LocalSystem`) than the interactive user who would naturally store the
credential — so an entry saved under your login is invisible to the service, and
email fails silently.

`notify.resolve_password()` documents storing the credential under the account
BlueIris actually runs as (`psexec -s -i` for `LocalSystem`), with a
`GMAIL_PASSWORD` environment fallback.

> **Lesson:** "works when I run it" and "works as a service" are different
> questions — decide who owns the credential up front.

## A raw event sum scores frame count, not activity

The first risk-scoring calibration replayed two weeks of real events and
produced **234 URGENT frames — every one a household burst.** BlueIris
re-triggers motion every ~30 s during sustained activity, so "sum the recent
events" made *one garage door standing open* look like twenty distinct signals.

The fix wasn't smaller weights, it was a different shape: **repeat dampening** —
the newest event of each camera+label counts in full, each older repeat counts
half again. Same data, re-replayed: **11 URGENT frames**, all genuine
multi-camera person sequences.

> **Lesson:** when a score misbehaves, ask what it is actually measuring — and
> replayable logs make the fix a five-minute experiment instead of another two
> weeks of live tuning.

## In the dark, the vision model guesses the alarming answer

The driveway gate label worked fine by day, then `GATE_OPEN_NIGHT` started
firing almost every night — 00:03, 01:35, 04:56 — while the gate sat closed.

The tell was in the descriptions: identical generic boilerplate ("The gate is
open, allowing a vehicle to drive through") with zero scene detail. **Hallucination
has a texture.** In IR the model can't see the latch, and a VLM never says "I
can't see" — it picks something, and *something skews dramatic*.

Fix: an explicit uncertainty default, in the camera prompt *and* the system
prompt — "if you cannot clearly see the condition, choose the normal-state
label; never pick the alarming option because you cannot see clearly."

> **Lesson:** for any alert label, define what the model should say when the
> evidence is invisible — otherwise it will invent the interesting case, and it
> will do it every night at 3 a.m.

## A new label can break the labels beside it, and no rewording fixes it

A front-yard camera kept describing a neighbourhood cat as "a person walking on
the sidewalk", so `ANIMAL` was added to give the cat somewhere to land.

It took **five prompt versions and five A/B runs**, and every one failed
differently:

| Version | Failure |
|---|---|
| v4 | blind to people on the sidewalk — three verified frames labeled `NONE` |
| v5 | sidewalk people placed *on the property*, plus three invented labels |
| v6 | both at once, including four frames labeled `APPROACHING_HOUSE` (alerting) whose own descriptions read "walking on the sidewalk, moving **away** from the house" |
| v7 | best run of the five, still dropping people to `NONE` |

One frame explained all four — labeled `NONE`, described as *"a person walking a
dog on a leash. The person is on the sidewalk…"*. `ANIMAL` was defined as an
animal *"with no person around it"*, and **a dog-walker is a person and an
animal**, so the two labels competed for the same frame and the model answered
neither. That yard has dog-walkers constantly. The back-patio camera carries
`ANIMAL` without trouble, because nobody walks a dog across the patio.

The label was dropped, not reworded. The config ended byte-identical to where it
started — and production never ran any of the five, because each was measured
before it shipped.

> **Lesson:** before adding a label, grep the descriptions in `events.db` for the
> scene it covers. If it co-occurs with a label you already have, you are not
> adding a category — you are splitting one, and both halves get less reliable.

> **Second lesson, cheaper:** the defect being chased was **cosmetic** — a cat
> logged under a label that never emails. Decide what a fix is worth before the
> first rewrite, not after the fifth.

## An invented label is a silent miss

The model occasionally answers with a label its prompt never defined
(`BIRD_FLYING`, `MOWING_LAWN` in normal operation; one bad prompt revision
produced three in a single 104-frame run). It matches no `alert_on` entry, so
the event is logged and dropped — **indistinguishable in the database from a
frame that correctly didn't alert.** One of those invented labels was
`PERSON_ON_PROPERTY`: exactly the case that should have emailed.

`labels.py` derives the permitted set from each prompt and tags offenders
`unknown-label`; `review.py --unknown` audits history for them.

> **Lesson:** when your failure mode and your success mode write the same row,
> add the column that tells them apart.

## The A/B tool was measuring the sampler, not the prompt

Two candidate prompts for a driveway camera were rejected in one evening, on
disagreement rates of 32% and 49% against the live config. Then the live config
was run against *itself* — same frames, same prompt on both sides. **It agreed
with itself on 75% of them.**

The noise floor was 25%, and neither rejection had measured anything. `Modelfile`
sets `temperature 0.1`, not 0, so a single classification is a *sample* of what a
prompt answers, and an A/B that runs each config once is comparing two samples.

What rescued the method was that the noise was not uniform:

| Label family | Unanimous across repeated draws |
|---|---|
| gate open/closed | 52% |
| person / vehicle | **100%** — 90 draws, zero variation |

All of the instability sat in one label. The gate is distant chain-link,
usually backlit or seen at an angle: a genuinely borderline call, where the model
was not being unreliable so much as being asked a question the image could not
answer. Occupancy labels never wavered, so the tool measured *those* perfectly
well — gate-state disagreements simply were not findings on that camera.

That reframed the fix. The unstable label was not a wording problem to solve, it
was a question worth deleting. Removing the gate labels took self-consistency
from **64% to 93%** and — because an always-visible state label had been crowding
out occupancy — recovered a vehicle label that had never once fired in 51,023
events.

`ab_prompt.py --repeat N` now draws each config N times per frame, reports
self-consistency broken down per label, and says outright whether the A-vs-B
difference clears the floor.

> **Lesson:** any evaluation that samples a stochastic model once per item is
> measuring the sampler as much as the change. Establish the floor first — and
> check whether it is *concentrated*, because one label that will not hold still
> is usually a bad question rather than a bad model.

## The model could judge it but not describe it

A driveway camera's `UNFAMILIAR_VEHICLE` label had never fired — 0 times in
57,948 events — through six prompt rewrites over three weeks. Every rewrite
assumed the model was seeing the car and picking the wrong label.

The test that settled it: edit the prompt to claim the household cars are "a red
sedan and a dark green hatchback", then show it a white pickup. **It still
answered `KNOWN_VEHICLE`, in 47 of 48 draws.** The colour/body-type comparison
was not happening at all. In a fifteen-label prompt, identity is one clause
among many, and `KNOWN_VEHICLE` had quietly become the model's word for "a
vehicle is present". Deleting the "if you cannot tell, choose `KNOWN_VEHICLE`"
fallback did not unlock it either — those frames just became `NONE`.

The fix was a second call asking only the identity question. The first version
asked for **attributes** — "answer with its colour and body type" — and compared
them to a configured list in Python, on the theory that a string comparison
cannot decline to run. The comparison ran perfectly and the attributes were
wrong: the household pickup came back `white suv`, and cropped, `silver sedan`,
both 4/4 unanimous. Two false alarms on the owner's own truck, stable ones, so
repeated draws could not rescue them.

Asking for the **verdict** instead — "is this one of OUR two vehicles? OURS or
NOT-OURS" — scored 6/6, 18/18 draws unanimous, across four vehicles that were
not the household's and two that were. And its free-text descriptions were still
wrong in exactly the same ways: it called a grey crossover "Black SUV" while
correctly answering NOT-OURS.

> **Lesson:** when a model is unreliable at *describing* something, don't
> conclude it is unreliable at *deciding* about it. Taking the intermediate
> representation and computing the decision yourself feels more rigorous and was
> measurably worse — it threw away the judgement and rebuilt it out of the least
> reliable part of the answer.

> **Second lesson:** the question that finally worked was the one asked alone. A
> capability can be absent from a fifteen-label prompt and present in a
> one-question prompt, with the same model and the same image.

## One undersized GPU, three different error messages

Swapping in a newer vision model turned a working pipeline into a two-hour
outage that changed its error message three times. First `request (4702 tokens)
exceeds the available context size (4096)`. Raise the context and it became
`CUDA error: out of memory`. Get past that and it became a wall of
`120,0xx ms` — the client timeout, to the millisecond.

Three symptoms, one cause, and it was never the model: **the alert frames are
4 MP**. At ~1,000 vision tokens per megapixel that is ~4,700 tokens per frame,
which overflows a default context, and the encoder buffer for an image that
size will not fit beside the weights on an 8 GB card — so two cameras firing
together OOM, and the queue behind them times out.

Fix: cap the long edge of the image *sent to the model* at 1600 px
(`MAX_IMAGE_EDGE`). Same frame on disk, same attachment in the email, ~2,400
tokens instead of ~4,700, and per-call latency went from ~45 s to under 10 s.

> **Lesson:** when the error message keeps changing as you fix things, you are
> walking down a resource ceiling, not fixing separate bugs. Find the input
> dimension that scales the cost — here, pixels — before tuning the knobs named
> in the error.

## A reasoning model answers in a field you are not reading

The candidate model returned HTTP 200, no error, no warning — and an empty
label on every frame. `"content": ""`, with 595 characters sitting in a
`"thinking"` field the script never looked at.

`parse_answer("")` returns `("", "")`. An empty label matches no `alert_on`
entry, so the pipeline logs the frame and emails nobody. It looks exactly like
a quiet afternoon.

Fix: `classify()` now asks `/api/show` what the model advertises and sends
`think: false` when `thinking` is in its capabilities — gated on capability
rather than hardcoded, because the incumbent model has no such field and the
A/B tool swaps between them mid-run.

> **Lesson:** a model upgrade can be strictly better at the task and still break
> the contract, because the *response shape* changed rather than the answer. A
> 200 is not a success — assert on the field you actually parse.

## The 8 GB card had one tenant, and I evicted it

Testing a 23 GB model on the box that runs production meant Ollama unloaded the
live model to make room, then thrashed. Twenty-five alerts failed during the
probe. Worse, the wreckage was *convincing*: the reloaded production model came
back with a smaller context, so real frames started failing — and that looked
exactly like a latent bug that had been there all along.

It was diagnosed as one, and written up as one, before the timestamps gave it
away: the failures started in the same minute as the first probe, and the
hourly error rate before it was 0–2.

> **Lesson:** on a single-GPU box the diagnostic *is* a deployment. Check for
> live traffic before the first probe, and when you find a "pre-existing" bug
> during an investigation, correlate its start time against your own first
> command before believing it.

## The deploy script broke on its own success check

The first deploy in this project that wasn't a single config file — a new
module, a modified `cam_watcher.py`, and a config — got a PowerShell script so
the copy order couldn't take alerting dark. The script copied all three files
correctly, then **crashed on the step that verifies the result**: its regex for
the pinned Python path in `cam_watcher.bat` anchored on `^\s*set\s+PYTHON=`, the
real file wrote it differently, and `$null.Matches.Groups[1]` threw.

Net effect: a changed production system and no verdict. The worst pair to hand
someone mid-deploy, produced by the very step meant to prevent it.

What worked instead needed nothing parsed. `cam_watcher` imports every module at
load and only *then* validates its arguments, so running the BlueIris wrapper
with no arguments exercises the whole import graph under the exact pinned
interpreter and working directory:

```
cmd /c cam_watcher.bat     ->  "ERROR - Usage: cam_watcher.py ..."
```

A usage error is the all-clear. A traceback means roll back.

> **Lesson:** a verification step that can itself fail is worse than none, and
> strictly worse when it runs after the irreversible part. Wrap it, give it a
> default, and never let it be the thing that throws.

> **Second lesson:** prefer a check that exercises the real entry point over one
> that reconstructs how the entry point works. The regex was a model of the
> `.bat`; running the `.bat` was the `.bat`.

## A hard-coded Python path in the `.bat` silently stopped the alerts

`cam_watcher.bat` pins a full path to `python.exe`, because BlueIris runs under
a service account where `python` isn't on `PATH`. At one point that path was
wrong and the wrapper stopped launching the script — and because **BlueIris
ignores the exit code**, nothing surfaced the failure. The household's alerting
just quietly went dark until someone noticed. (The exact trigger is lost to
history — this is a home project, not a postmortem culture.)

That outage is why `watchdog.py` exists.

> **Lesson:** a fire-and-forget integration needs a dead-man's switch — silent
> success and silent failure look identical from the outside.

## The fix that silenced the alarm it was fixing

The driveway camera was emailing about cars parked on the public street. Not a
perception failure — the model *said so itself* in the same sentence it raised
the alarm: "a silver SUV is parked in the driveway beyond the gate, which is not
our property." Seventeen vehicle events in three days, five of them emails, and
twelve more filed under `UNKNOWN_VEHICLE` — a label no prompt defines, which the
model had invented for itself.

That last detail is the diagnosis. The prompt offered no way to say "there is a
car, and it is not on our property". The only non-alerting answer was `NONE`,
and the model will not say *nothing* while a car is plainly in the frame — so it
scattered those frames across whatever labels were available, including the one
that wakes you up. A missing contrast label, exactly as an earlier entry here
describes.

So: add `STREET_VEHICLE`. Defined in the prompt, weight 0, deliberately **not**
in `alert_on`. Against a corpus of eighteen adjudicated frames it was a clean
sweep — all four emailing false alarms became `STREET_VEHICLE`, unanimous across
five draws, with the on-driveway vehicles and the `PERSON` frames untouched. By
every measurement aimed at the problem, it worked on the first attempt.

Then it was scored against a corpus built weeks earlier for a different
question: the same frames, but with the config lying about which cars belong to
the household, so that every household vehicle on the driveway is a stranger and
the alerting label *must* fire.

```
                        baseline config      with STREET_VEHICLE
UNFAMILIAR_VEHICLE         7/8 (88%)              0/8 (0%)
```

Zero. A stranger parked on the driveway came back `KNOWN_VEHICLE` — *ours* — in
four draws out of five. The new label had widened the space of comfortable
non-alarming answers, and the model slid into it and stopped raising alarms at
all. The camera would have gone quiet about the single event it exists for,
while every dashboard and count aimed at the reported bug looked excellent.

> **Lesson:** a change that fixes false alarms must be measured against a test
> that can only pass if real alarms still fire — and that test has to lie to the
> model, because the real event may never have happened on camera. The corpus
> that caught this took an evening to build and had already been "used up" on a
> different question months before.

## The camera could not see the thing the label was named for

`APPROACHING_HOUSE` on the front camera had fired 25 times and emailed 19. On
inspection, 15 of those were people on the **public sidewalk**, two were on the
property, and the rest were ambiguous. Six prompt revisions went into teaching
the model the boundary between a pavement and a garden path. Each one fixed one
label and broke another: the sidewalk cases, then the empty frames, then the
delivery courier, then the sidewalk cases again.

The seventh attempt was to ask the owner a question I should have asked first:
where is the camera pointed? It is mounted beside the front door, looking down
the side of the house. **The door and porch are not in frame.** In 67,000
events, the camera had never once produced a picture of a person at the door —
and a deliberate walk-to-the-door test produced five frames, every one of them
on the sidewalk, with a 41-second gap where the walkway walk should have been.

The label was named for an event this camera physically cannot observe. Its
"true positive" rate was not low, it was structurally zero, and every alert it
had ever sent was a false one. Meanwhile the approach path — the driveway — is
covered by a different camera whose `PERSON` label already includes it.

The fix was one line of config: remove the label from `alert_on`. It stays in
the prompt, so it is still produced and logged; it just no longer emails. Two
labels removed that way accounted for **83 of the camera's 201 lifetime emails**.

> **Lesson:** before tuning what a model says about a scene, establish what the
> camera can see. And when a label fires almost exclusively on the wrong thing,
> the lever is the alerting config — deterministic, immediate, and immune to the
> next model upgrade — not a seventh rewording.

## Two probes, opposite answers, no capability

An earlier entry here records the win from taking one question out of a crowded
fifteen-label prompt and asking it alone. That pattern was the obvious candidate
for the driveway region problem too, so before building any of the plumbing I
probed the isolated question over eleven adjudicated frames, five draws each.

Version one described the boundary as a gate. It answered "past the gate" for
everything near it — including vehicles parked at the far end of our own drive.
Version two described the driveway as one continuous strip, with the far end
explicitly ours. It then answered "on our drive" for **everything**, unanimously,
including a car on the road and a van in a neighbour's driveway.

Both scored 73%. Neither was measuring the vehicle's position; each was echoing
whichever side of the boundary the prompt had leaned on hardest.

Compare the probe from the time the pattern *did* work: 6 frames, 6 correct,
18 draws out of 18 unanimous. That is what a real capability looks like when you
isolate it. Two runs that fail in opposite directions are not a wording problem
to iterate on.

> **Lesson:** probe an isolated question at least twice, worded to lean opposite
> ways. If the answer follows your emphasis rather than the image, the model
> cannot do the task and no second call will rescue it — cost of finding out,
> twelve minutes; cost of not finding out, a module, a config schema and a test
> suite built on sand.

## The broken camera that looked like a broken prompt

For weeks the driveway camera had been producing occasional frames with the
bottom of the image replaced by flat green. It was logged as a cosmetic
annoyance on a wireless camera and left in the backlog.

It was not cosmetic. The missing region was the driveway itself — and shown a
frame whose lower two-thirds is a green rectangle, the model does not say it
cannot see. It describes a vehicle *"parked near the garage"*, precisely where
the picture ends, and the pipeline emails it.

```
frames 08-10 .. 08-19            truncation rate by day
UNFAMILIAR_VEHICLE   20 frames    08-10   6.0%     08-18  13.6%
  10 truncated (50%)              08-15  19.7%     08-19  24.3%
  17 emails, 8 from truncated
overall  1401 frames, 11.9%       after the fix:   0 / 988
```

Half the vehicle alerts in that window came from frames that did not contain the
driveway. Worse, it had been quietly poisoning the diagnosis of an unrelated
bug: of eighteen frames where a second-call identity check had overridden the
first answer, nine were truncated — so on half of them, the "wrong" decision was
being made about a vehicle that was never there.

The camera fault and the prompt fault produced the same symptom, in the same
label, in the same week. Fixing the wireless link removed as much of the noise
as the code change did, and the code change got all the credit until the frames
were measured.

> **Lesson:** when a model reports something impossible, check the input before
> debugging the reasoning. And an input-integrity check is worth building even
> for a fault you have already fixed — nothing in the pipeline could tell a
> corrupt frame from an empty driveway, so both wrote the same row and neither
> raised a flag.
