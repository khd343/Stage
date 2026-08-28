# RS-Stages — Locked Quantitative & Decision Specification

**Status:** LOCKED — v2.2  
**Date:** 2026-08-24  
**Repository:** `Pareshking/RS-Stages`  
**Primary technology:** Streamlit

> **Specification change:** The supplied **NSE Signal Interpretation Guide (Aug 2026)** is now the production specification for RS/Stage interpretation and Action decisions. The earlier v1.2 document was an engineering baseline and is superseded where this v2 document differs.

## 1. Authority hierarchy

1. Explicit project decisions in this v2 specification.
2. The supplied NSE Signal Interpretation Guide for interpretation/action rules.
3. Source books for their documented concepts: Weinstein stage structure, O'Neil
   RS/breakout principles, and — NEW v2.2 — Minervini's trend template and
   volatility-contraction pattern.
4. Clearly documented implementation assumptions.

Where a book states a criterion numerically, it is implemented verbatim and
attributed. Where a book describes a pattern qualitatively, any detector is an
RS-Stages operationalization, labelled as such, and never attributed to the
author as though it were their formula.

No implementation may silently invent a missing mathematical definition. If the guide names a condition without enough information to calculate it, the condition remains explicitly unavailable until operationalized and tested.

## 2. Universe and classification

- Production universe: official Nifty Total Market constituent CSV: `data/ind_niftytotalmarket_list.csv`.
- The official CSV is authoritative for the live constituent count; do not hard-code exactly 750.
- Industry is exactly the NSE CSV `Industry` field.
- No F&O filtering.
- No WealthStar sector remapping.
- Optional liquidity filtering never changes the mathematical RS ranking universe.
- Universe refresh remains the repository's scheduled Friday process.

## 3. Market data and information boundary

- Data source: yfinance.
- `auto_adjust=True`.
- Core price inputs: adjusted Close and adjusted High.
- Volume: raw/unadjusted.
- Decisions are pre-market for the upcoming NSE session.
- For decision session D, only information through the latest completed NSE session T may be used.
- No upcoming/incomplete-session data may enter calculations.
- Missing history produces explicit insufficiency; never fabricated values.

## 4. Relative Strength

- Lookbacks: 3, 6, 9 and 12 **calendar months**.
- Reference: last available NSE session on or before each calendar reference date.
- `R_period = Close_latest / Close_reference - 1`.
- Blend: `0.40×R3M + 0.20×R6M + 0.20×R9M + 0.20×R12M`.
- Cross-sectional score: `rank(Blend, pct=True, method='min') × 98 + 1`, rounded to integer 1–99.
- No skip month.
- Ranking occurs before optional liquidity UI filters.

### RS interpretation — NEW v2

- **80–99:** leadership.
- **50–79:** adequate, not leadership.
- **<50:** lagging.

The previous UI thresholds of RS 85/70 are retired.

## 4.1 The RS line — NEW v2.2

`RS_Score` is a cross-sectional percentile at one instant: it says where a stock
ranks today, not what its strength has been doing. The RS *line* supplies the
trajectory, and it is the one measure in this specification that can lead price.

Definition, on the shared completed-session calendar:

```
RS_Line(t)          = Close(t) / Benchmark_Close(t)
RS_Line_High_52W(T) = max RS_Line over the trailing 52 calendar weeks ending T
RS_Line_At_High(T)  = RS_Line(T) >= RS_Line_High_52W(T) * (1 - 0.005)
```

The benchmark is `^CRSLDX`, already locked in §12.2 as reference data. Using it
here does not promote it: no Stage, RS ranking or Action rule reads the RS line.

The 0.5% tolerance exists because "at a new high" compared with `==` on floating
point is a test that essentially never passes.

**The divergence — O'Neil's leading tell.** When relative strength reaches a new
high while price has not, the stock is outperforming inside its own base. That
ordering is the signal:

```
RS_Line_NH_Before_Price(T) = RS_Line_At_High(T) and Pct_From_52W_High(T) <= -5.0
```

The concept is O'Neil's and attributed to him. The −5% floor separating "price
has not yet made its high" from noise is an RS-Stages operationalization under
§1, not a number O'Neil states.

**Insufficiency.** The RS line requires benchmark closes across the full 52-week
lookback on the stock's own session calendar. Where the benchmark is missing for
a session, that session is excluded from the maximum; where fewer than 200
sessions of overlap exist, `RS_Line_High_52W` and both flags are unavailable and
published as such. No benchmark value is ever forward-filled, interpolated or
substituted to complete the window.

## 5. Stage — 30W MA

The 30W MA remains a **30-calendar-week SMA over all valid NSE sessions in the calendar window**, ending at T. It is not a fixed 150-row trading-day average.

Slope remains:

`Slope%(T) = (MA_30W(T) / MA_30W(T-10 sessions) - 1) × 100`.

Classification:

- Stage 2 — Advancing: Close > MA and slope > 0.
- Stage 3 — Topping: Close > MA and slope ≤ 0.
- Stage 4 — Declining: Close ≤ MA and slope ≤ 0.
- Stage 1 — Basing: Close ≤ MA and slope > 0.

Stage is categorical; never treat stage numbers as arithmetic quantities.

## 5.1 Minervini trend template — NEW v2.2

Stated numerically in the source and implemented verbatim. Eight criteria, each
published as its own boolean plus a 0–8 count, so a stock failing on one
criterion is distinguishable from one failing on six.

```
TT1  Close > SMA_150 and Close > SMA_200
TT2  SMA_150 > SMA_200
TT3  SMA_200 rising over the trailing 21 completed sessions
TT4  SMA_50 > SMA_150 and SMA_50 > SMA_200
TT5  Close > SMA_50
TT6  Close >= Low_52W * 1.30
TT7  Close >= High_52W * 0.75
TT8  RS_Score >= 70

Trend_Template_Score = count of TT1..TT8 satisfied
Trend_Template_Pass  = all eight satisfied
```

A session average is the mean of the latest N closes that **exist**. Sessions the
provider left empty are dropped before the window is taken, never averaged
around inside a fixed N-slot slice: that would report the mean of N-1
observations as an N-session average and hide the shortfall, which §3 forbids.
A calendar-window average needs no such rule because its bounds are dates, so a
gap inside it changes nothing.

**SMA_150 and SMA_200 are new session-based simple moving averages and are NOT
the existing `MA_30W`.** §5 locks the 30-week average as a *calendar-week*
construction, deliberately, and 30 calendar weeks is not 150 trading sessions.
Substituting one for the other would silently restate Minervini's criteria in
another author's units. They coexist; neither is redefined.

TT3 uses one month ≈ 21 completed sessions. The source prefers a longer
confirmation (four to five months) but states one month as the minimum, so the
minimum is what is tested and the stricter reading is left to the reader.

**Threshold provenance.** The 30% (TT6), 25% (TT7) and 70 (TT8) figures are
reproduced from the published trend template. They are transcribed, not derived,
and should be checked against the source text before this section is treated as
settled; every one of them is a single constant in `quant.py` for that reason.

## 6. 52-calendar-week high

- Use the preceding 52 calendar weeks ending at T.
- Minimum 200 valid sessions.
- `High_52W = max(adjusted High)` in the window.
- `Near_52W_High = Close_T >= 0.97 × High_52W`.

## 7. Volume

Prior-50-session baseline:

`Volume_MA50 = rolling(50, min_periods=50).mean().shift(1)`.

`Volume_Ratio = Volume_T / Volume_MA50`.

The latest completed session is the numerator; its volume is excluded from the baseline.

## 8. Up/Down volume

For each completed session:

- Close up → volume is Up Volume.
- Close down → volume is Down Volume.
- Unchanged → neither.

Use the 20 completed sessions ending at T, including T.

`U_D = UpVol20 / DownVol20`.

No arbitrary denominator offset is permitted. Zero-denominator handling remains explicit.

Interpretation:

- >1.5: Strong Accumulation.
- >1.3 to 1.5: Accumulating.
- 0.7 to 1.3: Neutral.
- <0.7: Distribution Warning.
- <0.6: Heavy Distribution.

## 8.1 Data integrity

- Calendar periods remain calendar based.
- The information boundary is global.
- The global boundary is the newest session carrying a Close for at least
  `100 - MAX_UNIVERSE_LOSS_PCT` percent of the universe, not the newest session
  that exists. Yahoo backfills the NSE universe over roughly 16-40 hours while
  NSE reopens 17.75 hours after a close, so a pre-market run can never see a
  complete latest session; chosen by recency the snapshot splits across two
  sessions and `RS_Score`, a cross-sectional percentile, ranks the fresher
  cohort against the staler one. Symbols without a Close at the boundary are
  excluded as explicit insufficiency; symbols the provider updated early are
  truncated to it. See DECISION_LOG D-2.2.16.
- Forward filling/interpolation requires explicit justification.
- Missing history produces explicit insufficiency.
- Optimizations require numerical regression testing against an independent/reference calculation.
- No performance improvement is a valid reason to change a locked mathematical definition.

## 9. Breakout

`Breakout` remains separate from `Breakout_Confirmed`.

Breakout setup:

- Stage 2.
- Close within 3% of 52W High.
- Volume Ratio >1.5×.

Confirmed breakout:

- Breakout setup.
- U/D >1.3.

The two states must never be collapsed.

## 9.1 Published snapshot fields — NEW v2.1

The audit publishes, in addition to the fields above:

- `Close` — the adjusted close of the latest completed session T. Every
  price-derived presentation value traces to this single observation.
- `Ext_Pct` = `(Close / MA_30W - 1) × 100`. This is the **displayed** extension.
  The locked `Extended_20Pct` condition keeps its specified form
  `Close > 1.20 × MA_30W` and is **not** re-derived from `Ext_Pct`: the two are
  algebraically equal but not bit-identical in floating point, and the
  comparison in section 10.1 remains the authority.
- `Pct_From_52W_High` = `(Close / High_52W - 1) × 100`.
- `Above_MA_30W`, `Above_MA_10W`, `MA10W_Above_MA30W`, `MA_30W_Rising` — strict
  comparisons between locked fields.
- `Trend_Health` — an integer 0–5, the count of the five conditions in
  section 9.3. It is a display aggregate; no locked signal consumes it.

## 9.2 The 10-calendar-week MA — NEW v2.1

`MA_10W` is a simple average over **every valid NSE session in a 10-calendar-week
window** ending at T, constructed exactly as the 30-week MA in section 5. It is
not a fixed 50-row trading-day average.

The alternative interpretation — reusing the 50-session `SMA_50` already
computed for the guide's below-50DMA condition — was considered and rejected:
placing a trading-day average beside a calendar-week average would make the two
trend lines non-comparable. `SMA_50` remains, unchanged, for its own condition
in section 10.2.

`MA_10W` is a trend reference and a checklist input only:

- it does **not** reclassify Stage;
- no locked signal, breakout condition or Action rule depends on it;
- the Stage definition in section 5 is unchanged.

## 9.3 Trend health — NEW v2.1

`Trend_Health` counts these five conditions, all locked fields or strict
comparisons between them:

1. `Close > MA_30W`
2. `MA_30W_Slope_10S_Pct > 0`
3. `MA_10W > MA_30W`
4. `Close > MA_10W`
5. `RS_Score >= 50` (the locked "not lagging" band from section 4)

## 9.4 52-week low — NEW v2.1

`Low_52W` is the minimum adjusted Low over the preceding 52 calendar weeks
ending at T, requiring at least 200 valid sessions — the same window and the
same guard as `High_52W` in section 6. It is a presentation/range input; no
locked signal consumes it. When the provider frame carries no `Low` column the
field is explicit insufficiency (NaN) and is never substituted with `Close`.

## 10. New guide-derived timing fields

### 10.1 Extension

The guide's **extended >20%** condition is operationalized as:

`Extended_20Pct = Close > 1.20 × MA_30W`.

This is a timing warning, not a Stage reclassification.

### 10.2 50DMA

The guide's **below 50DMA** condition is operationalized as the 50 completed-session simple moving average of Close:

`SMA50(T) = mean(Close over the latest 50 completed sessions)`.

`Below_50DMA = Close_T < SMA50(T)`.

This is a timing warning, not a Stage reclassification.

## 10.3 Pullback / volume drying — RESOLVED v2.2

Held open through v2.1: the guide named the condition without a definition
precise enough to calculate, so no detector was fabricated.

v2.2 closes it by sourcing the concept rather than inventing it. Minervini's
volatility-contraction pattern is the documented treatment of contraction plus
volume drying, and enters at authority level 3 alongside Weinstein and O'Neil.
See §10.5. The distinction §1 draws applies in full: the trend template is
numeric in the source and implemented verbatim; the contraction sequence is
described qualitatively and its detector is ours, labelled as ours.

## 10.4 Volatility — NEW v2.2

```
TrueRange(t) = max( High(t) - Low(t),
                    |High(t) - Close(t-1)|,
                    |Low(t)  - Close(t-1)| )
ATR_14(T)    = mean TrueRange over the 14 completed sessions ending T
ATR_Pct(T)   = ATR_14(T) / Close(T) * 100
```

Requires `Low`, which §9.1 already treats as optional in the provider frame.
Where `Low` is absent `ATR_Pct` is unavailable and published as such; no
high-minus-close proxy is substituted, because that is a different quantity.

## 10.5 Volatility contraction and volume dry-up — NEW v2.2

Closes the item §10.3 held open. The concept is Minervini's; **the detector below
is an RS-Stages operationalization**. The source describes a sequence of
successively tighter contractions with diminishing volume, illustrated by
example rather than reduced to a formula, so §1 requires the formalization be
labelled as ours and not presented as his.

Base window: the trailing 50 completed sessions, split into five consecutive
blocks of ten, oldest first.

```
Range_Pct(b)       = (max High in b - min Low in b) / mean Close in b * 100
VCP_Contractions   = count of adjacent pairs with Range_Pct(b[i+1]) < Range_Pct(b[i])
Contraction_Ratio  = Range_Pct(b[4]) / Range_Pct(b[0])
Volume_DryUp       = mean Volume over the last 10 sessions
                     / mean Volume over the 50 sessions preceding those 10
```

`Contraction_Ratio < 1` means the range is tightening; the source's rule of
thumb that each contraction runs roughly half the previous corresponds to a
ratio near 0.5 across the base. `Volume_DryUp < 1` means volume is drying.

```
VCP_Setup = Contraction_Ratio <= 0.60
            and Volume_DryUp <= 0.80
            and VCP_Contractions >= 2
```

Three thresholds, all ours, all single constants in `quant.py`.

This measures something `Volume_Ratio` cannot. §7's ratio compares one session
against a baseline and therefore detects the *spike* that confirms a breakout.
`Volume_DryUp` compares a sustained recent window against a longer one and
detects the *drought* that precedes it. They are deliberately opposite
instruments and neither replaces the other.

`Range_Pct` requires `Low`. Where it is absent the entire contraction group is
unavailable, `VCP_Setup` included; it is never computed from High and Close
alone.

## 10.5.1 Volatility contraction, from the source — NEW v2.3

> **STATUS: SPECIFIED, NOT IMPLEMENTED.** The detector described below has been
> built and has failed validation against real price history four times. It is
> not wired into the screener and nothing in this section is published. The
> Coiling screen continues to run on §10.5's detector, whose known defects are
> recorded there. Read this section as a design under test, not as settled
> specification. The validation record is in §10.5.2.

§10.5 above was written before the source text was available. Its detector was
an RS-Stages construction: five fixed ten-session blocks, compared by their
high-low range. The source specifies something different, and specifies it
precisely enough that almost none of the invention is needed. This section
supersedes §10.5's detector. §10.5's reasoning about *why* contraction matters
stands; its arithmetic does not.

### The base

The base begins at the absolute high the stock comes off and runs to T. Its
duration is **3 to 65 weeks**; outside that range the structure is not a VCP.

This replaces the fixed 50-session window, which was the single most damaging
choice in §10.5: of the six worked examples the source gives, five have bases
the old window cannot represent at all.

### Contractions

A contraction is a peak-to-trough decline measured from a swing high to the
following swing low:

```
Depth(i) = (SwingHigh(i) - SwingLow(i)) / SwingHigh(i) * 100
```

- Count: **2 to 6**, typically 2 to 4.
- Depths contract from left to right — this is the pattern's defining property.
- The deepest correction is **10% to 35%** in a constructive base.
- A deepest correction of **60% or more is rejected**: the source states such
  structures are prone to failure, and the overhead supply argument in §10.5
  explains why.

Note this measures *pullback depth*, not the range of a fixed window. The two
coincide only by accident.

### The technical footprint

The source publishes a base as three measurements, and so do we:

```
Base_Weeks      how long the base has been forming
Deepest_Pct     the largest correction anywhere in the base
Tightest_Pct    the narrowest pullback, at the far right of the base
Contractions    how many contractions (the source writes these as "T"s)
```

Rendered in the source's own shorthand: `40W 31/3 4T`.

### The pivot

Two cases, and the distinction is the source's:

1. **A base with real contractions** — the pivot is the high of the **final,
   narrowest** contraction. Not the high of the base.
2. **A flat base with no real contraction** — the pivot is the high of the base,
   and only when that base corrected no more than 10–15%.

Case 2 is what §10.6 currently implements for every base, which is why it is
right for flat bases and wrong for every VCP.

The pivot **may sit below the 52-week high** — the source names cup-with-handle
and cup-completion-cheat structures whose pivots form below the overall high.
It is therefore never gated on proximity to a new high.

### Volume

On the final contraction: average volume **below the 50-day average**, with at
least one session at or near the lowest volume in the entire base.

This supersedes `Volume_DryUp`'s comparison of the last ten sessions against the
prior fifty. That construction was ours and its 0.80 threshold was invented; the
rule above is the source's, and it is anchored to the final contraction rather
than to a fixed recent window.

### The one parameter that remains ours

Measuring pullbacks requires deciding what counts as a swing rather than noise.
The source reads this by eye and never states a threshold. The tightest
contraction in its worked examples is 2%, so the threshold must sit below that:
**1.5%** is used. This is the only invented number left in the section, against
a detector that was previously invented end to end.

### Acceptance criteria

The detector is not finished until it reproduces the source's own footprints:

| Stock | Footprint | Contractions |
| --- | --- | --- |
| MELI | `6W 32/6 3T` | 32 → … → 6 |
| New Oriental | `8W 22/2 3T` | 22 → 8 → 2 |
| FSII | `10W 18/5` | 18 → 5 |
| NFLX | `27W 27/7 3T` | 27 → … → 7 |
| VIVO | `40W 31/3 4T` | 31 → 17 → 8 → 3 |
| KCP | 4T | 32 → 14 → 7 → 3 |

These are acceptance tests against the source's own charts, not against our
universe. A detector that satisfies our data but not these is measuring
something else.

### Out of scope

Three further setups in the source are separate patterns, each with its own
entry criteria, and none is implemented or approximated here. All three are
recorded so a later revision can pick them up deliberately rather than
rediscovering them:

- **Power play / high tight flag** — a 100%+ advance in under eight weeks on
  heavy volume, then a sideways range correcting no more than 20–25% over three
  to six weeks, with volume contracting sharply just before the breakout. The
  source requires VCP characteristics *within* it, so it composes with §10.5.1
  rather than replacing it.
- **Primary base** — the first buyable base after an IPO: at least three to five
  weeks, correcting no more than 25–35%; a three-week consolidation should not
  correct more than 25%, while a base lasting around a year may decline as much
  as 50% and still be sound.
- **Cup-completion-cheat (3C) and cup-with-handle** — separate patterns with
  their own parameters — a 3 to 45 week formation, a cheat plateau contained
within 5–10%, a handle in the upper third of the cup, and a prior advance of
25–100% or more over the preceding 3 to 36 months. They are not implemented and
are not approximated by the VCP detector.

Post-entry management — squats, reversal recoveries, holding the 20-day average,
tennis-ball action — is position management, not screening, and is out of scope
for this specification entirely.

## 10.5.2 What validation against the source's own charts showed — NEW v2.3

Two of the source's worked examples are still listed with history covering the
period, so the detector can be measured against a reading made by the method's
author rather than against fixtures written here. That check is
`scripts/validate_vcp_footprints.py`, run in CI because the development
environment cannot reach the price provider.

It has been run four times against four detector designs. All four failed.

| Attempt | Design | MELI (`6W 32/6 3T`) | NFLX (`27W 27/7 3T`) |
| --- | --- | --- | --- |
| 1 | fixed 10-week window | not reached | not reached |
| 2 | fixed 1.5% threshold | `5W 29/4 6T` | `25W 17/6 26T` |
| 3 | cascading threshold | `10T`, deepest 23 | `44T` |
| 4 | cascade + structural bounds | no qualifying base | no qualifying base |
| 5 | recovery gating (synthetic only) | not reached | not reached |

Attempt 2 established that no single threshold can work: a sweep across eleven
values showed NFLX's deepest leg needs roughly 15% sensitivity while its
tightest needs 8% or finer, and the contraction count needs something between.
Every value failed differently.

Attempt 3 was worse than attempt 2. Its threshold ratcheted downward after every
contraction and never recovered, so late in a noisy base it degenerated into the
fine fixed threshold it was meant to replace.

Attempt 4 added the two properties the source states — contractions shrink left
to right, and the count is two to six — which fixed the degeneration on
synthetic data but not on real data. Both stocks now yield more than six
contractions, so the footprint is withheld. This fails safely rather than
reporting a fabricated count, but it does not read the pattern.

**Attempt 5 and why it stopped before real data.** The first four asked how
large a reversal must be to count. Attempt 5 asked instead what event *ends* a
contraction, and answered: price must retrace most of the decline before the
next one can begin, so counter-rallies inside a decline cannot fragment it. It
acts on the segmentation rather than filtering its output, which is the flaw
attempt 4 could not escape.

On a synthetic base built with realistic recoveries and daily noise — the two
properties the earlier fixtures lacked — it reproduced the built count of three
at every recovery fraction from 0.70 to 0.90, but only with a 3% noise floor.
The stability across recovery values is real evidence that the structural rule
carries the work. The floor is what kills it: the source's tightest reported
contraction is 2%, and the tightest leg is the one that forms the pivot, so a 3%
floor makes the pivot invisible by construction. At the 1-2% floors real
contractions demand, the count fragments to between five and twelve.

That is attempt 2's wall reached from a new direction. It failed on synthetic
data, so it was never run against MELI and NFLX.

**A correction to this section's earlier reasoning.** The pivot was recorded
here as the more tractable half of the problem, on the grounds that locating one
contraction is smaller than counting all of them. That is wrong and the error
matters. The pivot sits at the top of the *final* contraction, which in this
pattern is the *tightest* one — nearest the noise floor, where every detector
built here is weakest. The pivot is not the easy part of the count; it is the
part most exposed to the failure mode.

**What every attempt got right.** Base duration, deepest correction and tightest
correction landed within tolerance from attempt 2 onward. It is specifically the
**contraction count** that has never been reproduced, and the deepest reading
degrades only as a consequence of miscounting.

**The standing decision.** The contraction count is not published, and after
attempt 5 the pivot refinement is not pursued either. The elements
that do validate — the adaptive 3-to-65-week base, the depth bounds, the pivot
at the final contraction, the volume rule and the Stage 2 gate — are adopted;
the count and the `nW d/t nT` footprint that depends on it are not. A number
that cannot be reproduced against the source's own charts has no business in a
screen that claims to implement the source's method.

**A note on method, recorded because it recurred.** Each of the first three
attempts reached for a parameter where the source was describing a structure,
and each passed the full local test suite while being wrong. Fixtures written
alongside a detector can only confirm the assumptions both share. Every real
defect in this section was found by the external check.

## 10.6 The pivot — NEW v2.2

The buy point at the top of the base, and the distance still to travel.

```
VCP_Pivot    = max High over the trailing 50 completed sessions
Pct_To_Pivot = (VCP_Pivot / Close - 1) * 100
```

`Pct_To_Pivot` is zero when the stock is making the high of its own base and
negative once it has cleared it. This supersedes nothing: §6's `Near_52W_High`
remains a 3% boolean against the 52-week high, a different and longer reference.
A stock can sit on its base pivot while far below its 52-week high, which is
precisely the case a base-building screen must be able to see.

## 11. Production Action framework — NEW v2

The production Action vocabulary is now:

`BUY★`, `BUY`, `HOLD`, `WAIT`, `WATCH★`, `WATCH`, `REDUCE`, `SELL`, `AVOID`.

### Stage 4

**Always SELL**, regardless of RS.

### Stage 3

- RS <50 → SELL.
- RS ≥50 → REDUCE.

### Stage 1

- RS ≥80 → WATCH★.
- RS 50–79 → WATCH.
- RS <50 → AVOID.

### Stage 2 — RS ≥80

- Distribution (`U_D <0.7`) → REDUCE.
- Extended >20% → WAIT.
- Below 50DMA → WAIT.
- Confirmed breakout → BUY★.
- Breakout without confirmation → BUY.
- Otherwise → HOLD.

### Stage 2 — RS 50–79

- Distribution → REDUCE.
- Breakout/pullback without leadership → WAIT.
- Otherwise accumulating/holding → HOLD.

### Stage 2 — RS <50

**WAIT.**

The complete deterministic mapping is documented separately in `docs/ACTION_SPEC.md` and implemented in `rs_stages/actions.py`.

## 11.1 Stage 1 readiness — NEW v2.2

§11 assigns every Stage 1 stock the same instruction: wait for the breakout.
That is correct as an *action* and useless as a *ranking*. Stage 1 is the
largest bucket in most sessions and contains both the bases about to resolve
upward and stocks that are simply dead, and nothing published so far
distinguishes them.

`Stage1_Readiness` is a 0–5 count, defined only for Stage 1 rows and unavailable
elsewhere:

```
R1  MA_30W_Slope_10S_Pct >= -0.10     the decline has stopped
R2  RS_Score >= 50                    no longer lagging the universe
R3  Contraction_Ratio <= 0.70         the range is tightening
R4  Volume_DryUp <= 0.90              volume is drying
R5  Close > MA_10W                    the shorter line has been reclaimed
```

**No locked signal consumes this field.** It ranks a bucket for reading order;
it does not alter Stage, RS, breakout or any Action label, and a Stage 1 stock
scoring 5 still carries the Stage 1 action. This mirrors the standing of the
Signal Card bands in §12: presentation vocabulary over published numbers, never
a new decision rule smuggled in through a score.

## 12. Action transparency

Every Action shown to a user must expose:

- Action.
- Stage.
- RS score and RS band.
- 30W MA and slope.
- 52W High and proximity.
- Volume Ratio.
- U/D and distribution state.
- Breakout and Breakout Confirmed.
- 50DMA state.
- Extension state.
- Exact reason for the Action.
- Source/design attribution.

The Action is never permitted to hide the underlying mathematics.

## 12.1 Published artifacts — NEW v2.1

The audit downloads market data once and runs the identical pipeline at two
decision dates, publishing:

| Artifact | Contents |
| --- | --- |
| `data/latest_research.csv` | The snapshot at decision date D (latest completed session T). |
| `data/previous_research.csv` | The same pipeline with the boundary moved back one completed session (latest completed T-1). |
| `price_panel.npz` | A dense sessions x symbols grid of `Close` (float32) for the trailing 420 sessions, plus the session calendar and symbol list. **Published as a rolling release asset, never committed.** |
| `data/breadth_history.csv` | Point-in-time participation counts for the trailing 120 sessions. |

Constraints:

- Both snapshots must come from the same pipeline version. Diffing against a
  snapshot produced before a field existed would report the field's *arrival* as
  a market change, so any field missing from either side is skipped entirely.
- The price panel stores `Close` only. The moving averages are deliberately
  **not** stored: the presentation layer recomputes them for the single symbol
  it draws, using the same locked functions, so a drawn line cannot drift from
  the definition it claims to show.
- The breadth history is a stack of point-in-time counts. Each session's count
  uses only moving averages evaluated at that session, so the series carries no
  look-ahead. Symbols without a valid average at a session are excluded from
  both that session's numerator and its denominator.
- The panel is **never committed**. It is a regenerated binary that changes
  completely each run, so Git cannot delta it: measured cost is 1.43 MB of
  permanent history per run if committed, against 0 MB as a replaced release
  asset. It is published to the rolling `data-latest` release tag, whose single
  asset is overwritten every run.
- The panel is stored as a compressed NumPy grid rather than Parquet. Every
  symbol shares the same completed-session calendar, so a dense matrix is both
  smaller (measured 0.88 MB against 1.39 MB) and readable with NumPy alone. The
  presentation layer therefore requires no Arrow runtime to draw a chart.
- Because the panel and the committed snapshot are published to different
  places, they can drift. The audit refuses to publish a panel whose terminal
  session disagrees with the snapshot's decision date, and the presentation
  layer withholds a mismatched panel rather than drawing it. A chart and a table
  must never describe different sessions.
- The panel is loaded lazily and held by reference, never serialised into the
  snapshot cache. Only the two views that draw price history load it, so a
  failure to read it degrades those two rather than the whole terminal.

## 12.2 Benchmark index — NEW v2.1

A benchmark index is published alongside the breadth history purely as a
reference line on the Market page.

- Ticker: `^CRSLDX` (Nifty 500), fetched with the same locked adjustment policy.
- It is **not** part of the analytical universe. No RS ranking, Stage
  classification, breakout condition or Action rule reads it.
- Index tickers bypass `yfinance_symbol`, which maps NSE constituent symbols and
  must stay locked to that job.
- The index tracks 500 companies while breadth tracks the whole Nifty Total
  Market universe. A divergence between the two lines can therefore be
  composition rather than market behaviour, and the chart says so.
- Breadth is a percentage and the index is a price level, so they are drawn on
  separate axes. Sharing one scale would flatten whichever has the smaller
  range.
- A failure to fetch the index never fails the audit: breadth is ours and
  computed, the index is an external convenience. The column is simply absent
  and the page reports it.

## 12.3 Breadth window and coverage — NEW v2.1

The breadth history retains 250 completed sessions. A 30-week average needs
roughly 150 sessions of warm-up, so the oldest part of a long window is
measurable for only a few symbols. Sessions where fewer than 50% of the panel
has a valid moving average are excluded: a percentage taken over a handful of
stocks is not breadth, and plotting it would show a spike that means nothing.

## 13. Liquidity

Liquidity remains a UI/screener filter only:

`AvgValue20 = mean(Close × raw Volume over latest 20 completed sessions)`.

Liquid when `AvgValue20 > ₹5 crore`.

RS ranking is never recomputed after applying this filter.

## 14. UI specification — NEW v2

The platform is a **quantitative research product**, not a plain Streamlit form.

Required information architecture:

1. **Dashboard** — market snapshot, stage breadth, action distribution, strongest setups, recent movers.
2. **Screener** — dense sortable table with Action as the final decision column, filters for Stage/Action/RS/Industry/Liquidity, search, and clear evidence fields.
3. **Industries** — industry leadership, breadth and action concentration.
4. **Movers** — strongest RS changes and stage/action transitions where data supports them.
5. **Stock** — professional individual-stock research page with TradingView Lightweight Charts, 30W MA overlay, decision evidence and Action card.
6. **Methodology** — plain-language formulas, source attribution, information boundary and Action rules.

Visual direction:

- White/light neutral canvas.
- Strong typography hierarchy.
- Subtle green/blue/amber/red accents with semantic meaning.
- Compact professional tables.
- No oversized meaningless numeric cards.
- Numbers must be formatted for human reading: RS as integer, percentages as percentages, volume ratio with ×, U/D to sensible precision, INR with Cr/L notation.
- Mobile-first responsive behavior.
- TradingView Lightweight Charts is a charting library/component only; it is not the source of quantitative calculations.

## 15. Superseded v1 behaviour

The following early design decisions are retired:

- Five-label Action system (`BUY/HOLD/WAIT/REDUCE/SELL`).
- RS ≥85 BUY threshold.
- RS ≥70 HOLD threshold.
- Action logic embedded directly inside `app.py`.
- UI that presents raw decimal returns/values without human formatting.
- UI whose Action explanation is less detailed than the underlying evidence.

The quantitative RS/Stage definitions remain unchanged unless explicitly modified above.

## 16. v2.1 change summary

Additive only. No v2.0 definition was altered:

- Added `MA_10W`, `Low_52W`, `Close`, `Ext_Pct`, `Pct_From_52W_High`,
  `Trend_Health` and the trend-health booleans to the published snapshot.
- Generalised the calendar-window moving average into a single shared
  definition. `ma_30w` and `ma_30w_series` now delegate to it and are proven
  bit-identical to the previous implementation, including on gapped history.
- Added the previous-session snapshot, the price panel and the breadth history
  as published artifacts.
- Extended the independent reconciliation in the audit to the 10-week MA, the
  52-week low and the price panel.

RS, Stage, the 52-week high, volume ratio, U/D, breakout, confirmation, the
timing warnings, liquidity and the nine-label Action framework are byte-for-byte
unchanged.


## 17. v2.2 change summary

Additive only. No v2.0 or v2.1 definition was altered.

- Added §4.1 (the RS line and its divergence), §5.1 (the Minervini trend
  template), §10.4 (ATR), §10.5 (volatility contraction and volume dry-up),
  §10.6 (the base pivot) and §11.1 (Stage 1 readiness).
- Added Minervini to the §1 authority hierarchy, with the rule that a numeric
  criterion is implemented verbatim and attributed while a qualitative pattern
  gets a detector labelled as ours.
- Resolved §10.3, held open through v2.1 for want of a precise definition, by
  sourcing the concept rather than inventing one.
- Added 25 published snapshot fields. None is read by any decision rule: Stage,
  RS ranking, the breakout tests and all nine Action labels are byte-for-byte
  unchanged, and a test compares every v2.1 column with and without the new
  benchmark input to prove it.

Two corrections were forced by the first scheduled audit run, both older than
v2.2 and neither previously triggered:

- §3 — the information boundary is the latest session carrying a Close, not the
  latest row the provider emitted. A dated row published before its values are
  final is not a completed session, and adopting one shifted every calendar
  window a session late.
- §5.1 — a session-count average is the mean of the latest N closes that exist.
  Averaging whatever survives inside a fixed N-slot slice reported the mean of
  N-1 observations as an N-session average. Calendar-window averages are
  unaffected, being bounded by dates.

The trend-template thresholds in §5.1 remain transcribed rather than verified
against the source text, and are labelled provisional wherever they surface.
