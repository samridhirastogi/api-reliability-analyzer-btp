# AI-based reliability scoring of backend APIs using runtime data

B.Tech project — Samridhi Rastogi

Scores every backend API in a microservice system on a 0–1 reliability scale
from its runtime telemetry, combining a learned failure-risk estimate with
measured error frequency and response-time stability, and ranks the APIs so the
failure-prone ones surface early.

**Every measurement in this project was recorded from a running system.** No
latency, error rate, request rate or resource reading is simulated — each one is
what a real microservice application produced under a real workload generator.

The faults were deliberately injected in a controlled testbed rather than waited
for in production, and that is the point rather than a concession: it is what
makes the labels exact. The benchmark records precisely when each fault began
and which service it hit, so the ground truth comes from the injection log and
never from the telemetry the model is scored on.

---

## Where the data comes from

The pipeline runs on **RCAEval**, a published benchmark of real microservice
telemetry:

| | |
|---|---|
| Source | `phamquiluan/RCAEval` (WWW'25 companion / FSE'26), Hugging Face |
| Systems | Online Boutique, Sock Shop, Train Ticket — three open-source microservice applications |
| Recordings | 735 failure cases; **733 usable** after validation |
| Telemetry | per-second p50/p90 latency, error rate, request rate, CPU, memory, disk I/O, sockets, per service |
| Ground truth | the exact Unix timestamp at which a real fault was injected, and which service it was injected into |
| Fault types | CPU exhaustion, memory pressure, disk stress, network delay, packet loss, socket exhaustion, and five code-level faults |

Each case is a recording of one of those applications running under a real
workload generator while a real fault was injected into one service. The
measurements are what the running system actually produced; the benchmark's
injection log is what supplies the labels, independently of the telemetry.

Two of the 735 cases are rejected automatically: one has a corrupted injection
timestamp and one is a truncated 62-second recording. `python -m src.ingest`
prints the validity report.

## Results at a glance

733 recorded cases → **816,822 scored windows** over 47 APIs; 393,136 of those
are unambiguous enough to train and test on, 8.5 % of them failing.

| | |
|---|---|
| PR-AUC on held-out **cases** | **0.956** (a trivial baseline scores 0.088) |
| ROC-AUC / F1 / Brier | 0.990 / 0.916 / 0.0116 |
| 5-fold CV grouped by case | PR-AUC 0.964 ± 0.002 |
| Held-out faults caught | **218 / 220**, median **29 s** after injection |
| Mean reliability score, healthy windows | **0.982** |
| Mean reliability score, failing windows | **0.443** |
| Healthy windows called "Reliable" | 99.9 % |
| Failing windows called not-Reliable | 92.2 % |

A PR-AUC of 0.956 against a 0.088 base rate is high, and it should be: this is
detection of an API that is *already* in a failing state, not prediction of an
unprovoked future failure (finding 1). Strong separability is what a working
detector looks like on this task. The demanding numbers are the transfer results
in finding 2, and the labelling sensitivity check below.

`reports/RESULTS.md` has the full tables, `reports/ABLATIONS.md` the two
ablations, `reports/BASELINES.md` the baseline comparison, and
`reports/PROPAGATION.md` the measurement behind the labelling choice. Figures
land in `reports/figures/`. The write-up — literature survey, research gap,
methodology — is in [`docs/THESIS.md`](docs/THESIS.md).

### Against baselines, and under a harder labelling

Same features, same training cases, same held-out cases (`reports/BASELINES.md`):

| Model | Main labelling | Propagation-negative |
|---|---|---|
| **Random forest** | **0.963** | **0.935** |
| Logistic regression | 0.748 | 0.614 |
| CPU level alone | 0.623 | 0.561 |
| Latency z-score alone | 0.120 | 0.055 |
| *PR-AUC floor (base rate)* | *0.085* | *0.041* |

A single metric is not enough, a linear combination of all of them is not
enough either, and the value is in the interactions — that is the case for
building a model rather than tuning a threshold.

(The forest reads 0.963 here against 0.956 in the table above. Same split and
same seed; the headline model gives up a quarter of its training cases to an
isotonic calibration holdout, which costs a little raw ranking accuracy and buys
a risk output that can honestly be read as a probability. These baselines skip
calibration so that every model in the comparison sees identical training data.)

The second column is the sensitivity check on the labelling decision. The main
pipeline masks out services degraded by a fault in a *dependency*, because the
benchmark does not record which callers ended up degraded; the alternative
forces those windows into the negative class instead.

**The two columns are not directly comparable** — PR-AUC is measured against the
positive rate, and that halves from 0.085 to 0.041 when the propagating windows
join the negative class, so the columns have different floors. Against their own
floors the model is 11× chance under the main labelling and **23× under the
alternative**. On either reading the result survives: absolute PR-AUC falls only
0.963 → 0.935, while the linear and single-metric baselines degrade much
further, so the harder labelling widens the model's advantage rather than
narrowing it. The masking decision is not what is carrying the headline number.

### Three findings worth defending in the viva

**1. It detects, it does not foresee — and 29 s is a windowing result, not a
model result.** Faults are injected at externally chosen moments, so nothing in
the preceding telemetry predicts them. What the model does is notice a failing
state fast. With 30 s windows on a 15 s hop, the first window that can contain
*any* post-injection sample ends 14 s after injection, and the observed delays
are quantised to exactly that grid: 14 s, 29 s, 44 s, … **27 % of cases are
flagged in the very first window that could possibly show the fault, 89 % by the
second.** The median of 29 s is two windows — near the mechanical floor, so the
model is not the bottleneck. Finer windows would cut the delay at the cost of
noisier features; that trade-off, not model capacity, sets how early this fires.

**1b. Unseen fault types split along a clean line.** Every fault whose mechanism
is itself an input feature transfers well (memory 0.982, CPU 0.981, socket
0.915) — a model that never trained on a memory fault still watches memory
climb. Faults with no such channel, visible only as latency, are where it
struggles: packet loss 0.623, and network delay which reaches 0.852 PR-AUC but
recovers only **32 % of the fault**. Disk stress sits between at 0.802, which
fits the same explanation — disk I/O is a feature, but exported for only about a
third of services. (Code-level faults are *not* the weak family, contrary to the
obvious guess: f2 scores 0.932. The f4/f5 rows have 19 and 6 cases and carry
wide error bars.)

**2. It does not transfer to an unseen application.** Trained on two of the
three systems and tested on the third, PR-AUC falls from 0.956 to 0.54–0.80.
The ablation in `reports/ABLATIONS.md` tested the obvious explanation — that the
model leans on absolute CPU and memory levels, which are deployment-specific —
and **refuted it**: restricting the model to scale-free features transfers
*worse* (0.445 vs 0.669), while absolute features alone nearly match the full
set. The honest conclusion is that the model should be re-fitted on a new
system's own telemetry rather than dropped in untrained.

**3. The failure-risk term does nearly all the work.** Ranking windows by the
calibrated risk alone scores 0.981; the full weighted score scores 0.966. Error
frequency (0.133) and response consistency (0.145) are weak discriminators on
their own, largely because only ~27 % of services export an error metric and
most faults in this benchmark degrade latency without returning errors. The
composite score still earns its place — it tells an operator *why* an API scored
0.37, which a bare probability does not — but its value is explanation rather
than ranking power, and the 0.40 / 0.35 / 0.25 weighting is best defended as a
presentation decision, not an accuracy optimisation.

## Quick start

```bash
pip install -r requirements.txt

python run_pipeline.py            # download → features → labels → train → score → report
streamlit run dashboard/app.py    # open the dashboard
```

The download is about 310 MB and the full run takes roughly 30–45 minutes on a
laptop. For a faster first pass:

```bash
python run_pipeline.py --per-group 1     # one run per fault group instead of five
```

Individual stages can be re-run on their own:

```bash
python -m src.ingest        # corpus summary / validity report (writes nothing)
python -m src.download      # fetch recorded cases
python -m src.features      # 1 Hz telemetry -> causal per-window features
python -m src.labels        # attach ground-truth failure labels
python -m src.train         # fit + evaluate the failure-risk model
python -m src.score         # compute reliability scores
python -m src.report        # figures and RESULTS.md
python -m src.endpoints     # optional: score individual endpoints from spans
python -m src.ablation      # optional: the two ablations
python -m src.baselines     # optional: baselines + labelling sensitivity
python -m src.propagation   # how far faults actually spread
python -m src.weights       # validate the score's weights (OECD method)
```

`python -m src.baselines` compares against the alternative labelling, which is
built with
`python -m src.labels --propagation-negative --out data/processed/labelled_propneg.parquet`.

`python -m src.endpoints` needs raw request spans, which are large and not
downloaded by default. Fetch them for a handful of cases with
`python -m src.download --with-traces 6`.

## How a score is produced

**1. Windowing.** Telemetry is cut into 30-second windows starting every 15
seconds. One reliability score is produced per API per window.

**2. Features (32 of them).** Each window carries *level and shape* features —
mean, spread and maximum latency, the p90/p50 tail ratio, coefficient of
variation, error rate, request rate, CPU and memory — and the same quantities
*relative to that API's own preceding 90 seconds*. The intent of the relative
features is that a catalogue lookup and a checkout call have very different
normal latencies, so the model should react to an API *changing* rather than to
an API being slow.

They help, but not in the way that was expected: the ablation shows they do not
carry the cross-system signal on their own, and the absolute levels turn out to
matter more than anticipated. See finding 2 above.

Every feature is computed from samples at or before the window end, and the
trailing baseline stops one full window short of the current window, so the two
never share a sample. Nothing from the labelled horizon can leak backwards.

**3. Failure risk.** A random forest (400 trees) estimates the probability that
the API is in a failing state during the next 30 seconds. That horizon is
forward-looking, but it overlaps the fault being detected rather than reaching
ahead of it, so what the model delivers is fast detection of a failing state and
not advance warning — see finding 1. The output is
**isotonically calibrated** on a separate split of training cases, because the
score treats the risk as a probability and weights it at 0.40 — a raw forest
vote share would make that weight meaningless.

**4. Reliability score.**

```
Reliability = 0.40 · (1 − FailureRisk)
            + 0.35 · (1 − ErrorFrequency)
            + 0.25 · ResponseConsistency
```

Bands: **≥ 0.70 Reliable**, **0.50–0.70 Degraded**, **< 0.50 Failure-prone**
(the proposal's worked example, 0.73, stays "Reliable").

### Two deliberate changes from the proposal

**Response consistency.** The proposal defines it as the inverse of the standard
deviation of latency. That is unbounded — a perfectly steady API gives 1/0 — and
unit-dependent: the same API scores differently when latency is recorded in
milliseconds rather than seconds. It therefore cannot be added to two terms that
live in [0, 1]. The implementation keeps the intent and uses the coefficient of
variation:

```
consistency = 1 / (1 + σ/μ)
```

which is dimensionless, sits in (0, 1], equals 1 for perfectly steady latency
and falls towards 0 as jitter grows.

**Unmeasured components.** Not every service in these systems exposes an
error-rate metric. Substituting zero would reward an API for being unmeasured,
so the missing term is dropped and the remaining weights are renormalised to sum
to one. The dashboard marks which APIs this applies to.

## How it is evaluated

Everything is **grouped by recorded case**: all windows of one recording sit on
the same side of every split, so the model is never tested on windows from a
recording it has already partly seen. Window-level splits would leak badly here,
because consecutive windows of the same case overlap by 15 seconds.

The report covers four evaluations:

- **Held-out cases** — 30 % of recordings kept aside entirely.
- **5-fold cross-validation grouped by case.**
- **Leave-one-system-out** — train on two applications, test on the third the
  model has never seen. This is the honest test of whether the approach
  transfers rather than memorising service names.
- **Leave-one-fault-type-out** — train without a fault type, test on it.

Alongside classification metrics, the report gives **detection delay**: how many
seconds pass between the real injection and the first window in which the
injected API is flagged.

### A note on labelling

The benchmark records which service a fault was injected into, not which
services ended up degraded — and a fault in one service propagates to everything
calling it. Labelling those callers "healthy" while they are visibly degraded
would teach the model to ignore real degradation, so during an active fault the
non-injected services are **masked out**: neither positive nor negative.
Negatives come from the pre-fault period. That makes the two classes easier to
separate than they would be in production, so the alternative labelling is not
just offered but **run**: see `reports/BASELINES.md`, where forcing the
propagating windows into the negative class costs only 0.963 → 0.935 PR-AUC.

## What the dashboard shows

`streamlit run dashboard/app.py`

- **Live scoring** — replay a recording window by window and see what an on-call
  engineer would have been shown at each moment: every API ranked by reliability
  score, and a breakdown of which component is dragging each score down.
- **Case timeline** — reliability score across the whole recording with the real
  injection marked, over the raw telemetry underneath it.
- **Model** — held-out performance, detection delay, generalisation to unseen
  systems, and which measurements the model relies on.
- **Method** — the scoring definitions, in the dashboard itself.

A toggle restricts the case list to recordings held out of training, so a demo
can be run entirely on data the model has never seen.

## Layout

```
src/config.py      every constant that affects a result
src/download.py    fetch recorded cases from the benchmark
src/ingest.py      normalise per-suite schemas; validity checks
src/features.py    windowing and causal feature extraction
src/labels.py      ground-truth labels from the injection log
src/train.py       model, calibration, four evaluations
src/score.py       the reliability score
src/report.py      figures and RESULTS.md
src/ablation.py    feature-transfer and score-component ablations
src/baselines.py   logistic/threshold baselines + labelling sensitivity
src/propagation.py measures how far a fault actually spreads
docs/THESIS.md     the write-up: survey, gap, methodology, results
src/endpoints.py   endpoint-level scoring from raw request spans
src/viz.py         shared palette and chart theme
dashboard/app.py   Streamlit dashboard
data/raw/          downloaded recordings (gitignored)
data/processed/    windows, labels, scores
models/            fitted model + metrics.json
reports/           RESULTS.md, figures, run logs
```

## Limitations

- Faults are injected at externally chosen moments, so the model detects a
  failing state quickly rather than predicting an unprovoked failure minutes
  ahead. The reported detection delay is the honest measure of how early it is.
- The scored unit is a service (one backend API surface). Endpoint-level
  scoring would need per-endpoint spans, which the benchmark provides for a
  subset of cases.
- Error rate is exported by only ~27 % of services, so the 0.35 term is
  renormalised away for most APIs. The ablation shows the failure-risk term
  carries almost all the ranking power; the error and consistency terms serve as
  explanation of *why* a score fell, not as independent signal.
