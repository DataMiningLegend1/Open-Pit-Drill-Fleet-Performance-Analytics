# Open-Pit Drill Fleet Performance Analytics

## Executive summary

The source contains **749 holes**, **5 rigs**, **6 operators**, and **10 dates**. Total recorded length is **11,172.5 m**.

After transparent screening, **663 records** were eligible for modelling. There are **50 missing rates**, **10 domain-limit outliers above 250 m/hr**, and **36 robust statistical outliers**. All remain in the cleaned export with flags.

Quality-filtered median penetration rate is **42.00 m/hr**. Redrill rate is **3.60%** and wet-hole rate is **3.20%**.

## Operational highlights

- Highest median-rate rig: **SDR_07** at **52.50 m/hr**.
- Highest median-rate operator: **Mulenga Stanley** at **53.55 m/hr**.
- Consensus anomalies requiring engineering review: **27**.

## Statistical evidence

- D'Agostino normality: p=7.331e-38
- Kruskal-Wallis by equipment: p=2.455e-21
- Welch ANOVA by equipment: p=1.033e-14
- Kruskal-Wallis by operator: p=4.838e-14
- Welch ANOVA by operator: p=5.003e-11

## Predictive modelling

Best regression model: **ElasticNet**; temporal holdout MAE **9.61 m/hr**, RMSE **12.91 m/hr**.

Best low-performance classifier: **ExtraTrees**. Metrics are saved in `models/classification_metadata.json`. Threshold selection uses out-of-fold training probabilities only; the temporal holdout remains untouched until final evaluation.

## Engineering recommendations

1. Review consensus anomalies before correcting or deleting any record.
2. Compare operators within rig and hole-type context, not using raw averages alone.
3. Treat model risk as a prioritization signal, not an automated personnel judgement.
4. Add shift, start/finish time, delays, maintenance state, bit condition, geology, planned depth, and cost fields.
5. Monitor drift by date and validate findings with supervisors and operators.

## Limitation

This dataset lacks explicit plans, delay codes, maintenance, cost, and direct rock-hardness measurements. Results demonstrate analytics competence and association, not causal proof or an autonomous dispatch recommendation.
