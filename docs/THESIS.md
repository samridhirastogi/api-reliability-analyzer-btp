# AI-Based Reliability Scoring of Backend APIs Using Runtime Data

**B.Tech Project — Samridhi Rastogi**

---

> **Note on references.** Every work cited in Section 3 is listed in full in
> Section 12. Two have been checked against their publisher records
> (Zhou et al. 2021; Saito & Rehmsmeier 2015). The remainder are standard
> references in their areas, but **verify each one against the published record
> before submission** — author lists, venues and years should not be taken on
> trust from this document.

---

## 1. Problem Statement

A backend system of any size is a collection of APIs that depend on one
another. When one of them starts to degrade — responses slow down, errors
appear, latency becomes erratic — the symptoms surface across the system, and
the engineer on call has to decide which API is actually in trouble. Current
practice answers this with per-metric thresholds: alert when p99 latency
exceeds some value, or when the error rate crosses a percentage.

Thresholds are a poor fit for the problem. They are set per service by hand,
they do not transfer between services with different normal behaviour, and
they treat each metric independently when the informative signal is usually a
*combination* — latency rising while CPU also rises means something different
from latency rising on its own. Most importantly, a threshold produces a binary
alert, not a graded answer, so it cannot rank a set of APIs by how unreliable
each one currently is.

This project builds a pipeline that takes runtime telemetry from a running
system and produces, for every API and every 30-second window, a single
reliability score between 0 and 1, combining:

1. a learned **failure risk** estimated by a machine-learning model,
2. the measured **error frequency**, and
3. the **response consistency** of the API's latency.

The APIs are then ranked by that score, so the failure-prone ones surface
without anyone configuring a threshold.

## 2. Objectives

1. Build a reproducible pipeline from raw runtime telemetry to a per-API
   reliability score.
2. Train a model that estimates, from recent telemetry alone, the probability
   that an API is in a failing state.
3. Combine that estimate with directly measured runtime metrics into one
   interpretable score, and validate the combination rather than assuming it.
4. Evaluate honestly: on recordings the model has never seen, on applications
   it has never seen, and on fault types it has never seen.
5. Establish what the approach *cannot* do, with measurements rather than
   speculation.

## 3. Literature Survey

**Classical software reliability.** Traditional reliability engineering models
software as a system with a failure rate, summarised by measures such as mean
time between failures, and fitted with growth models such as those of Musa or
Goel–Okumoto [1]. These assume a relatively static artefact whose failure
behaviour can be characterised over a long observation period. A microservice
API deployed continuously and serving fluctuating traffic does not fit that
assumption: its reliability is a property of the current runtime conditions,
not a constant to be estimated once.

**Service-level objectives and error budgets.** Modern operational practice
replaces a single reliability figure with service-level indicators and
objectives — an explicit target such as "99.9% of requests complete under
300 ms" — together with error budgets tracking the shortfall [2]. This is the
right granularity for runtime reliability, and this project's score is in the
same spirit, but SLOs are still configured per service by a human, and the
threshold must be chosen before anything can be measured against it.

**Microservice fault studies and benchmarks.** Zhou et al. conducted an
industrial survey of microservice debugging practice and released
**TrainTicket**, a benchmark system with a catalogue of reproducible injected
faults, finding that practitioners are limited by weak trace analysis and that
more intelligent analysis of runtime data is needed [3]. TrainTicket, together
with Sock Shop and Google's Online Boutique, has become the standard testbed
for this area, and all three are used in this project.

**Root-cause analysis from telemetry.** A substantial literature learns from
runtime telemetry to identify *which* service caused an incident. Nezha
correlates metrics, logs and traces to localise fine-grained root causes [4];
broader surveys catalogue anomaly-detection and root-cause methods for
microservice applications [5]. **RCAEval** [6] is the most directly relevant:
it packages 735 recorded failure cases from the three benchmark applications
above, with per-second telemetry and the ground-truth timestamp and target of
every injected fault, specifically to make such methods comparable. This
project uses RCAEval as its data source, but — importantly — not for the task
the benchmark was designed to evaluate (see Section 4).

**Production trace characterisation.** Large-scale traces released by operators,
notably Alibaba's microservice call-graph datasets [7], characterise dependency
structure and latency behaviour at production scale. These were evaluated as a
data source for this project and rejected: they carry no error or status field
and no fault labels, so neither the error-frequency component of the score nor
any supervised label could be derived from them.

**Fault injection and chaos engineering.** Deliberately injecting faults into a
running system to observe its response is established practice [8]. It is the
only practical way to obtain labelled failure data without waiting for
production outages, and it is what makes the ground truth in RCAEval exact.

**Machine-learning methods used.** The failure-risk model is a random forest
[9], chosen for its tolerance of correlated and differently scaled features and
its interpretable impurity-based importances. Because the score treats the
model's output as a probability and weights it, the raw forest vote share is
mapped to a calibrated probability by isotonic regression [10]. Evaluation uses
precision–recall rather than accuracy, as the positive class is a small
minority; the precision–recall plot is the more informative view under class
imbalance [11], although recent work argues the ROC curve retains more value in
this setting than that result is often taken to imply [12], so both are
reported here.

## 4. Research Gap

Three gaps motivate this work.

**(a) Reliability scoring is not root-cause analysis, and the literature
conflates them.** The RCA literature, and the benchmarks built to evaluate it,
ask *which service caused this incident* — a single correct answer per
incident. Operationally, a different question is at least as useful: *how
reliable is each API right now*, graded and ranked. These differ in a way that
matters for labelling. An API whose latency has doubled because its dependency
is failing is **not** the root cause, but it **is** currently unreliable from
its own consumers' point of view. No existing benchmark labels this second
question directly. Section 6.4 measures the size of that mismatch rather than
assuming it away.

**(b) Multi-metric reliability scores are asserted, not validated.** Composite
scores that combine latency, errors and other signals with fixed weights appear
in both tooling and literature, but the weights are typically chosen by
intuition and never tested. This project ablates every term of its own score
and reports the result even though it is unflattering (Section 9.3).

**(c) Cross-system generalisation is rarely measured.** Models in this area are
usually trained and tested on the same application, so it is unclear whether
they learn transferable behaviour or memorise one deployment. This project
evaluates leave-one-application-out and leave-one-fault-type-out, and reports
the drop.

## 5. Research Methodology

The pipeline has six stages, each a separate module so that any stage can be
re-run alone.

**5.1 Data acquisition.** Recorded failure cases are downloaded from RCAEval.
Each case contains per-second telemetry for every service in one application,
plus the Unix timestamp of a real fault injection and the service it targeted.

**5.2 Normalisation and validation.** The three benchmark suites export
different column names — one uses `_load` and a single `_latency`, the others
`_workload` with `_latency-50` and `_latency-90` — so all are mapped to one
schema. Each recording is validated: the injection timestamp must fall inside
the recording and the recording must be long enough to contain both a normal
and a faulty period. Two of the 735 cases fail these checks (one corrupted
timestamp, one truncated 62-second recording) and are rejected automatically.
Datastores, brokers and exporters are excluded from the scored set, as they are
not request-serving APIs.

**5.3 Windowing and feature extraction.** Telemetry is divided into 30-second
windows starting every 15 seconds; one score is produced per API per window.
Windows are built by aggregating non-overlapping 15-second blocks and combining
consecutive pairs, which makes overlapping windows exact and cheap to compute.

Each window carries 32 features in two families:

- *Level and shape*: mean, standard deviation and maximum of p50 latency;
  p90 latency and the p90/p50 tail ratio; coefficient of variation; error rate;
  request rate; CPU, memory, socket and disk measures.
- *Relative*: each of the above expressed as a ratio and a z-score against the
  same API's own preceding 90 seconds.

The relative family exists because absolute levels are not comparable across
services: a catalogue lookup and a checkout call have very different normal
latencies, so the model should respond to an API *changing* rather than to an
API being slow.

**Leakage control.** Every feature is computed from samples at or before the
window end, and the trailing baseline stops one full window short of the current
window, so the two never share a sample. No information from the labelled
horizon can reach the feature vector.

**5.4 Labelling.** Labels come from the benchmark's injection log, never from
the telemetry — a label derived from latency would be circular, since latency is
also an input feature. A window is positive when its API is the one the fault
was injected into and the fault is active during the next 30 seconds.

The treatment of *other* services during a fault is a genuine methodological
decision, addressed in Section 6.4.

**5.5 Failure-risk model.** A random forest of 400 trees is trained on the
windowed features. Missing values are median-imputed with explicit missingness
indicators, so that "this service exports no error metric" stays distinguishable
from "its error rate was measured as zero". The forest's output is calibrated by
isotonic regression fitted on a separate split of training recordings, because
the reliability score multiplies this value by 0.40 and therefore needs it to
mean an actual probability.

**5.6 Reliability score.**

```
Reliability = 0.40 × (1 − FailureRisk)
            + 0.35 × (1 − ErrorFrequency)
            + 0.25 × ResponseConsistency
```

Bands: ≥ 0.70 Reliable, 0.50–0.70 Degraded, < 0.50 Failure-prone.

Two changes were required to make the proposal's formula well-formed:

*Response consistency.* The proposal defines it as the inverse of the standard
deviation of latency, 1/σ. That quantity is unbounded — a perfectly steady API
gives 1/0 — and unit-dependent, so the same API scores differently when latency
is recorded in milliseconds rather than seconds. It cannot be added to two terms
that live in [0, 1]. The implementation preserves the intent and uses the
coefficient of variation:

```
Consistency = 1 / (1 + σ/μ)
```

which is dimensionless, lies in (0, 1], equals 1 for perfectly steady latency,
and falls towards 0 as jitter grows.

*Unmeasured components.* Not every service exports an error-rate metric.
Substituting zero would reward an API for being unmeasured, so the missing term
is dropped and the remaining weights are renormalised to sum to one.

## 6. Dataset and Evaluation Setup

**6.1 Source.** RCAEval [6], obtained from Hugging Face
(`phamquiluan/RCAEval`), approximately 310 MB for the metric recordings.

| | |
|---|---|
| Applications | Online Boutique, Sock Shop, Train Ticket |
| Recorded failure cases | 735 downloaded, **733 usable** |
| Telemetry | per-second p50/p90 latency, error rate, request rate, CPU, memory, disk I/O, sockets |
| Ground truth | injection timestamp and target service |
| Fault types | CPU, memory, disk, network delay, packet loss, socket exhaustion, and five code-level faults |

**6.2 On the use of injected faults.** Every measurement in this dataset was
produced by a real application running under a real workload generator; none of
it is simulated. The faults were deliberately injected in a controlled testbed
rather than waited for in production. This is a strength rather than a
concession: it is what makes the labels exact, since the injection log records
precisely when each fault began and which service it hit, independently of the
telemetry the model is scored on. The corresponding limitation — that the model
detects rather than predicts — is stated in Section 10.

**6.3 Evaluation protocol.** All splits are **grouped by recorded case**: every
window of one recording falls on the same side of every split. A window-level
split would leak heavily, because consecutive windows of the same recording
overlap by 15 seconds. Four evaluations are reported:

1. **Held-out cases** — 30% of recordings reserved entirely.
2. **5-fold cross-validation**, grouped by case.
3. **Leave-one-application-out** — train on two applications, test on the third.
4. **Leave-one-fault-type-out** — train with a fault type removed, test on it.

Alongside classification metrics, **detection delay** is reported: the time
between the real injection and the first window in which the model flags the
injected API.

**6.4 The labelling decision, and how it was settled.** When a fault is injected
into one service, the services calling it may also degrade. The benchmark
records only which service was *injected*, so a caller that ends up degraded has
no ground-truth label. Two policies are possible:

- **Masked**: treat non-injected services during an active fault as neither
  positive nor negative, and exclude them.
- **Propagation-negative**: label them negative.

Rather than choose on intuition, the propagation rate was measured directly
(`src/propagation.py`): for every service in every recording, mean p50 latency
during the fault was compared with the same service's latency beforehand. The
result is reported in Section 9.4. Both policies were then trained and
evaluated, and the gap between them bounds what the choice is worth. This
measurement is used only to justify a policy; it never produces a label.

## 7. Tools and Scripts

Python 3.12. Libraries: `pandas` and `pyarrow` (data), `scikit-learn` (model,
calibration, evaluation), `matplotlib` (figures), `streamlit` and `altair`
(dashboard), `requests` (data acquisition).

| Module | Responsibility |
|---|---|
| `src/config.py` | every constant that affects a result |
| `src/download.py` | fetch recorded cases from the benchmark |
| `src/ingest.py` | normalise per-suite schemas; validity checks |
| `src/features.py` | windowing and causal feature extraction |
| `src/labels.py` | ground-truth labels from the injection log |
| `src/train.py` | model, calibration, four evaluations |
| `src/score.py` | the reliability score |
| `src/propagation.py` | measures how far a fault actually spreads |
| `src/baselines.py` | linear and single-metric baselines; labelling sensitivity |
| `src/ablation.py` | feature-family and score-component ablations |
| `src/endpoints.py` | endpoint-level scoring from raw request spans |
| `src/report.py` | figures and results tables |
| `dashboard/app.py` | interactive dashboard |

Run:

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
```

## 8. Dashboard

The dashboard replays a recording window by window and shows what an on-call
engineer would have been shown at each moment: every API ranked by reliability
score, a breakdown of which component is dragging each score down, the score
timeline against the real injection point, the raw telemetry underneath it, and
the model's held-out performance. A toggle restricts the case list to recordings
held out of training, so a demonstration can be run entirely on data the model
has never seen.

## 9. Results

> Numeric results are generated by the pipeline into `reports/RESULTS.md`,
> `reports/BASELINES.md`, `reports/ABLATIONS.md` and `reports/PROPAGATION.md`,
> with figures in `reports/figures/`. Section 9 summarises them; those files are
> the authoritative record and regenerate on every run.

See `reports/RESULTS.md` for the full tables.

## 10. Limitations

1. **Detection, not prediction.** Faults are injected at externally chosen
   moments, so nothing in the preceding telemetry can anticipate them. The
   model recognises a failing state quickly; it does not foresee an unprovoked
   future failure. The reported detection delay is the honest measure, and it is
   bounded below by the window geometry rather than by model capability.
2. **Limited cross-application transfer.** Performance drops substantially when
   the model is tested on an application absent from training. The model should
   be re-fitted on a new system's own telemetry rather than deployed untrained.
3. **The composite score is for explanation, not ranking power.** The ablation
   shows the calibrated failure risk alone ranks slightly better than the
   weighted combination. The score earns its place by telling an operator *why*
   an API scored as it did, but the proposal's weights are best defended as a
   presentation decision rather than an accuracy optimisation.
4. **Error frequency is sparsely measured.** Only about a quarter of services
   export an error-rate metric, so that term is renormalised away for most APIs.
5. **Service granularity.** The scored unit is a service. Endpoint-level
   scoring is implemented (`src/endpoints.py`) but only for the subset of
   recordings that ship raw request spans.
6. **Residual label ambiguity.** Distinguishing an API failing on its own from
   one dragged down by a dependency requires the call graph, which is available
   for only part of the corpus.

## 11. Future Work

1. **Call-graph-aware labelling.** Use parent–child span relationships to
   separate *failing at source* from *degraded by dependency*, turning the
   ambiguity of Section 6.4 into a three-class target. A preliminary check on
   six recordings was inconclusive — the services that transitively call the
   injected one showed no degradation, while a few that cannot reach it did, and
   only nine service-level edges were recovered — so this is a lead rather than
   a result.
2. **Weighting by discriminative power.** The three score components separate
   the two states by very different margins. Setting weights from measured
   separation, rather than by intuition, would close the gap between the
   composite score and the risk term alone.
3. **Endpoint-level scoring at scale**, by acquiring spans for the full subset
   of recordings that provide them.
4. **Online adaptation**, so that a model re-fits to a new deployment's
   telemetry automatically, addressing Limitation 2.

## 12. References

[1] J. D. Musa, *Software Reliability Engineering: More Reliable Software,
Faster and Cheaper*, 2nd ed., AuthorHouse, 2004. (See also A. L. Goel and
K. Okumoto, "Time-dependent error-detection rate model for software reliability
and other performance measures," *IEEE Transactions on Reliability*, 1979.)

[2] B. Beyer, C. Jones, J. Petoff and N. R. Murphy (eds.), *Site Reliability
Engineering: How Google Runs Production Systems*, O'Reilly Media, 2016.

[3] X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, W. Li and D. Ding, "Fault Analysis
and Debugging of Microservice Systems: Industrial Survey, Benchmark System, and
Empirical Study," *IEEE Transactions on Software Engineering*, vol. 47, no. 2,
pp. 243–260, 2021. DOI: 10.1109/TSE.2018.2887384. *(Verified.)*

[4] G. Yu et al., "Nezha: Interpretable Fine-Grained Root Causes Analysis for
Microservices on Multi-Modal Observability Data," *ESEC/FSE*, 2023.

[5] J. Soldani and A. Brogi, "Anomaly Detection and Failure Root Cause Analysis
in (Micro)Service-Based Cloud Applications: A Survey," *ACM Computing Surveys*,
2022. (arXiv:2105.12378)

[6] L. Pham, H. Ha and H. Zhang, "RCAEval: A Benchmark for Root Cause Analysis
of Microservice Systems with Telemetry Data," *Companion Proceedings of the ACM
Web Conference (WWW Companion)*, 2025. DOI: 10.1145/3701716.3715290.
arXiv:2412.17015. Dataset: https://huggingface.co/datasets/phamquiluan/RCAEval

[7] S. Luo et al., "Characterizing Microservice Dependency and Performance:
Alibaba Trace Analysis," *ACM Symposium on Cloud Computing (SoCC)*, 2021.
Dataset: https://github.com/alibaba/clusterdata

[8] A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski, J. Reynolds
and C. Rosenthal, "Chaos Engineering," *IEEE Software*, vol. 33, no. 3, 2016.

[9] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32,
2001.

[10] B. Zadrozny and C. Elkan, "Transforming Classifier Scores into Accurate
Multiclass Probability Estimates," *ACM SIGKDD*, 2002.

[11] T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative
than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,"
*PLoS ONE*, vol. 10, no. 3, e0118432, 2015. DOI: 10.1371/journal.pone.0118432.
*(Verified.)*

[12] E. McDermott et al., "The receiver operating characteristic curve
accurately assesses imbalanced datasets," *Patterns*, 2024. — included because
it argues against [11]; cite both if the choice of metric is discussed.
