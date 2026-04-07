# Qualitative Model Analysis

Generated: 2026-04-04T23:20:36
Lookback days: 14
Integrated data: True
Primary model for case breakdown: GRU

## Quantitative Context

```
 model  accuracy  precision   recall       f1  tp  fp   tn  fn
   GRU  0.683235   0.521298 0.311893 0.390281 257 236 1475 567
Hybrid  0.687574   0.535714 0.291262 0.377358 240 208 1503 584
```

## Sample Outputs (Most Interpretable Cases)

The following samples show concrete model outputs and put performance in context. Cases are selected by highest confidence distance from threshold.

### TP examples

```
 beach_id           beach_name forecast_date  actual  prediction probability confidence_distance  month_last  observation_count_last  diameter_cm_last  sting_last
       17      Palmahim-Rishon    2019-07-13       1           1       0.865               0.365         7.0                    14.0          6.250000         1.0
        7            Dor-Atlit    2019-07-16       1           1       0.847               0.347         7.0                    14.0          6.545455         1.0
       14     Tel Aviv-Herzlia    2017-06-25       1           1       0.837               0.337         6.0                    10.0          9.200000         1.0
       15       Jaffa-Tel Aviv    2017-06-26       1           1       0.811               0.311         6.0                    12.0         15.818182         1.0
       19      Ashkelon-Ashdod    2019-07-19       1           1       0.808               0.308         7.0                     4.0          3.000000         1.0
       14     Tel Aviv-Herzlia    2017-06-29       1           1       0.800               0.300         6.0                     8.0          0.875000         0.0
       14     Tel Aviv-Herzlia    2017-07-01       1           1       0.799               0.299         6.0                    10.0          8.666667         1.0
       10     Michmoret-Hadera    2019-07-20       1           1       0.796               0.296         7.0                     4.0          0.000000         0.0
       15       Jaffa-Tel Aviv    2017-06-20       1           1       0.794               0.294         6.0                     8.0         25.250000         1.0
       10     Michmoret-Hadera    2019-07-27       1           1       0.794               0.294         7.0                     8.0         13.666667         1.0
       17      Palmahim-Rishon    2015-07-15       1           1       0.793               0.293         7.0                     8.0         24.000000         1.0
       14     Tel Aviv-Herzlia    2021-07-04       1           1       0.787               0.287         7.0                    40.0         15.307693         1.0
       15       Jaffa-Tel Aviv    2015-07-04       1           1       0.786               0.286         6.0                     2.0         27.500000         1.0
       18      Ashdod-Palmahim    2017-07-02       1           1       0.785               0.285         7.0                    16.0         22.875000         1.0
       12        Gaash-Natanya    2015-05-22       1           1       0.784               0.284         5.0                     2.0         11.000000         0.0
       15       Jaffa-Tel Aviv    2016-07-04       1           1       0.783               0.283         7.0                    10.0          8.400000         1.0
        9  Hadera-Jisr a zarqa    2022-07-21       1           1       0.777               0.277         7.0                     4.0         20.000000         1.0
       14     Tel Aviv-Herzlia    2012-06-09       1           1       0.775               0.275         6.0                     2.0         27.500000         0.0
       12        Gaash-Natanya    2015-07-19       1           1       0.773               0.273         7.0                     6.0         32.833332         1.0
       19      Ashkelon-Ashdod    2022-06-27       1           1       0.769               0.269         6.0                    18.0          5.411765         1.0
        1 Nahariya-Rosh Hniqra    2015-07-17       1           1       0.765               0.265         7.0                     4.0         21.250000         1.0
       19      Ashkelon-Ashdod    2022-07-15       1           1       0.764               0.264         7.0                     2.0         26.000000         1.0
       17      Palmahim-Rishon    2022-07-21       1           1       0.763               0.263         7.0                     8.0         29.500000         0.0
       17      Palmahim-Rishon    2017-06-23       1           1       0.759               0.259         6.0                     4.0         13.750000         1.0
       18      Ashdod-Palmahim    2024-03-11       1           1       0.757               0.257         3.0                    14.0        104.178574         0.0
```

### TN examples

```
 beach_id           beach_name forecast_date  actual  prediction probability confidence_distance  month_last  observation_count_last  diameter_cm_last  sting_last
       17      Palmahim-Rishon    2020-08-26       0           0       0.069               0.431         8.0                     2.0               0.0         0.0
       11    Natanya-Michmoret    2021-08-20       0           0       0.073               0.427         8.0                     2.0               0.0         0.0
        7            Dor-Atlit    2017-09-25       0           0       0.073               0.427         9.0                     4.0               5.5         0.0
        5        Tira-Shiqmona    2017-09-21       0           0       0.074               0.426         9.0                     2.0               0.0         0.0
       19      Ashkelon-Ashdod    2021-08-26       0           0       0.074               0.426         8.0                     4.0               0.0         0.0
       17      Palmahim-Rishon    2021-08-20       0           0       0.074               0.426         8.0                     2.0               0.0         0.0
        5        Tira-Shiqmona    2015-09-24       0           0       0.074               0.426         9.0                     2.0               0.0         0.0
       12        Gaash-Natanya    2023-10-10       0           0       0.076               0.424         9.0                     2.0               0.0         0.0
       12        Gaash-Natanya    2022-10-07       0           0       0.077               0.423        10.0                     2.0              11.0         1.0
        7            Dor-Atlit    2020-09-25       0           0       0.077               0.423         9.0                     2.0               0.0         0.0
        9  Hadera-Jisr a zarqa    2017-08-05       0           0       0.078               0.422         8.0                     2.0               0.0         0.0
       14     Tel Aviv-Herzlia    2016-09-19       0           0       0.078               0.422         9.0                     4.0               0.0         0.0
       13        Herzlia-Gaash    2021-09-17       0           0       0.078               0.422         9.0                     2.0               0.0         0.0
       17      Palmahim-Rishon    2023-08-20       0           0       0.079               0.421         8.0                     2.0               0.0         0.0
       17      Palmahim-Rishon    2023-09-21       0           0       0.079               0.421         9.0                     2.0               0.0         0.0
        6           Atlit-Tira    2021-09-14       0           0       0.080               0.420         9.0                     2.0               0.0         0.0
        7            Dor-Atlit    2016-08-28       0           0       0.080               0.420         8.0                     2.0               0.0         0.0
        7            Dor-Atlit    2017-08-29       0           0       0.081               0.419         8.0                     2.0               0.0         0.0
        7            Dor-Atlit    2019-08-28       0           0       0.081               0.419         8.0                     2.0               0.0         0.0
       13        Herzlia-Gaash    2021-09-03       0           0       0.081               0.419         9.0                     2.0               0.0         0.0
        7            Dor-Atlit    2019-08-26       0           0       0.082               0.418         8.0                     2.0               0.0         0.0
        9  Hadera-Jisr a zarqa    2020-08-29       0           0       0.082               0.418         8.0                     2.0               0.0         0.0
        1 Nahariya-Rosh Hniqra    2023-08-11       0           0       0.082               0.418         8.0                     2.0               0.0         0.0
        3      Kiryat Yam-Acco    2020-09-11       0           0       0.082               0.418         9.0                     2.0               0.0         0.0
        5        Tira-Shiqmona    2017-08-30       0           0       0.082               0.418         8.0                     2.0               0.0         0.0
```

### FP examples

```
 beach_id           beach_name forecast_date  actual  prediction probability confidence_distance  month_last  observation_count_last  diameter_cm_last  sting_last
       16         Rishon-Jaffa    2015-07-15       0           1       0.834               0.334         7.0                     8.0         38.375000         1.0
       18      Ashdod-Palmahim    2012-06-13       0           1       0.816               0.316         6.0                     2.0         27.500000         0.0
       10     Michmoret-Hadera    2017-06-30       0           1       0.810               0.310         6.0                    10.0         23.400000         1.0
       15       Jaffa-Tel Aviv    2022-07-24       0           1       0.786               0.286         7.0                     6.0         30.666666         1.0
       19      Ashkelon-Ashdod    2015-06-21       0           1       0.784               0.284         6.0                     2.0         23.500000         1.0
       17      Palmahim-Rishon    2021-06-29       0           1       0.778               0.278         6.0                    20.0         12.722222         1.0
        7            Dor-Atlit    2019-07-14       0           1       0.768               0.268         7.0                    16.0         16.000000         1.0
       12        Gaash-Natanya    2015-05-31       0           1       0.763               0.263         5.0                     2.0         27.500000         0.0
       19      Ashkelon-Ashdod    2023-07-13       0           1       0.753               0.253         7.0                     2.0          0.000000         1.0
       10     Michmoret-Hadera    2019-07-19       0           1       0.749               0.249         7.0                     6.0          9.400000         1.0
       12        Gaash-Natanya    2021-07-12       0           1       0.746               0.246         7.0                     6.0          5.200000         1.0
       10     Michmoret-Hadera    2016-06-22       0           1       0.745               0.245         6.0                     6.0          8.666667         1.0
       18      Ashdod-Palmahim    2022-12-19       0           1       0.744               0.244        12.0                     6.0          5.000000         0.0
       17      Palmahim-Rishon    2021-06-27       0           1       0.736               0.236         6.0                    18.0         17.416666         0.0
       18      Ashdod-Palmahim    2022-12-29       0           1       0.736               0.236        12.0                     4.0          5.000000         0.0
       17      Palmahim-Rishon    2022-09-16       0           1       0.727               0.227         9.0                    14.0         41.454544         0.0
        5        Tira-Shiqmona    2019-07-26       0           1       0.727               0.227         7.0                     2.0         47.500000         1.0
       17      Palmahim-Rishon    2015-07-10       0           1       0.726               0.226         7.0                     4.0          5.500000         1.0
        1 Nahariya-Rosh Hniqra    2013-07-08       0           1       0.725               0.225         7.0                     2.0          0.000000         0.0
       11    Natanya-Michmoret    2019-07-19       0           1       0.723               0.223         7.0                     4.0         15.000000         1.0
       17      Palmahim-Rishon    2015-07-25       0           1       0.717               0.217         7.0                     6.0          4.800000         0.0
       17      Palmahim-Rishon    2017-07-04       0           1       0.713               0.213         7.0                     2.0         47.500000         1.0
        8     Jisr a zarqa-Dor    2024-04-27       0           1       0.706               0.206         4.0                     2.0       5001.500000         0.0
       19      Ashkelon-Ashdod    2021-06-26       0           1       0.701               0.201         6.0                    14.0         15.090909         1.0
       12        Gaash-Natanya    2017-07-06       0           1       0.699               0.199         6.0                     2.0         35.000000         1.0
```

### FN examples

```
 beach_id          beach_name forecast_date  actual  prediction probability confidence_distance  month_last  observation_count_last  diameter_cm_last  sting_last
       14    Tel Aviv-Herzlia    2019-09-23       1           0       0.070               0.430         8.0                     2.0          0.000000         0.0
        7           Dor-Atlit    2016-08-19       1           0       0.081               0.419         8.0                     2.0          0.000000         0.0
        3     Kiryat Yam-Acco    2021-08-15       1           0       0.082               0.418         8.0                     2.0          0.000000         0.0
        7           Dor-Atlit    2017-09-29       1           0       0.082               0.418         9.0                     2.0          0.000000         0.0
        3     Kiryat Yam-Acco    2021-09-20       1           0       0.083               0.417         9.0                     2.0          0.000000         0.0
        6          Atlit-Tira    2021-09-19       1           0       0.083               0.417         9.0                     2.0          0.000000         0.0
       14    Tel Aviv-Herzlia    2019-12-04       1           0       0.088               0.412        10.0                     2.0          0.000000         0.0
        4 Shiqmona-Kiryat yam    2020-09-13       1           0       0.089               0.411         9.0                     2.0          0.000000         0.0
        5       Tira-Shiqmona    2021-09-16       1           0       0.093               0.407         9.0                     2.0          0.000000         0.0
        8    Jisr a zarqa-Dor    2015-09-03       1           0       0.094               0.406         9.0                     2.0          0.000000         0.0
       10    Michmoret-Hadera    2022-09-12       1           0       0.094               0.406         9.0                     2.0          0.000000         0.0
       18     Ashdod-Palmahim    2021-09-21       1           0       0.096               0.404         9.0                     2.0          5.000000         0.0
        7           Dor-Atlit    2017-08-22       1           0       0.096               0.404         8.0                     4.0          0.000000         0.0
        7           Dor-Atlit    2023-09-16       1           0       0.096               0.404         9.0                     2.0          0.000000         0.0
        9 Hadera-Jisr a zarqa    2020-10-10       1           0       0.097               0.403        10.0                     2.0         20.000000         0.0
        3     Kiryat Yam-Acco    2021-10-29       1           0       0.098               0.402        10.0                     2.0          0.000000         0.0
       19     Ashkelon-Ashdod    2020-10-14       1           0       0.100               0.400        10.0                     2.0          0.000000         0.0
       10    Michmoret-Hadera    2022-10-12       1           0       0.103               0.397        10.0                     2.0          0.000000         0.0
        7           Dor-Atlit    2022-10-07       1           0       0.103               0.397        10.0                     4.0          0.000000         0.0
       13       Herzlia-Gaash    2021-08-05       1           0       0.104               0.396         8.0                     6.0          0.000000         0.0
       12       Gaash-Natanya    2020-10-28       1           0       0.104               0.396        10.0                     2.0         34.000000         0.0
        7           Dor-Atlit    2018-08-19       1           0       0.104               0.396         8.0                     6.0          2.833333         0.0
        2      Acco-Nahariyah    2021-10-24       1           0       0.105               0.395        10.0                     4.0          0.000000         0.0
       14    Tel Aviv-Herzlia    2023-07-14       1           0       0.106               0.394         7.0                     4.0          0.000000         0.0
        5       Tira-Shiqmona    2021-10-25       1           0       0.108               0.392        10.0                     2.0          0.000000         0.0
```

## Model Disagreement Cases

These are examples where GRU and Hybrid produce different class predictions, which helps explain strengths/weaknesses by input type.

```
 beach_id           beach_name forecast_date  actual  gru_prob  hybrid_prob  prob_gap
     13.0        Herzlia-Gaash    2019-07-27       0  0.259200     0.648331  0.389131
      8.0     Jisr a zarqa-Dor    2019-09-26       1  0.213119     0.598179  0.385060
     17.0      Palmahim-Rishon    2022-09-16       0  0.727317     0.351142  0.376176
     18.0      Ashdod-Palmahim    2012-06-13       0  0.815646     0.441207  0.374438
      3.0      Kiryat Yam-Acco    2022-07-25       1  0.252791     0.621912  0.369121
      1.0 Nahariya-Rosh Hniqra    2013-07-08       0  0.724577     0.363604  0.360973
     14.0     Tel Aviv-Herzlia    2012-06-09       1  0.774894     0.431237  0.343657
     10.0     Michmoret-Hadera    2023-12-02       0  0.581886     0.251122  0.330763
     19.0      Ashkelon-Ashdod    2019-12-22       1  0.699360     0.378402  0.320958
     16.0         Rishon-Jaffa    2015-08-01       0  0.238446     0.550663  0.312217
      1.0 Nahariya-Rosh Hniqra    2018-09-25       1  0.249120     0.557302  0.308182
     13.0        Herzlia-Gaash    2019-07-22       1  0.282363     0.590531  0.308169
     18.0      Ashdod-Palmahim    2012-05-12       1  0.612209     0.310717  0.301492
     19.0      Ashkelon-Ashdod    2024-02-04       1  0.541854     0.244038  0.297816
     18.0      Ashdod-Palmahim    2017-06-19       0  0.401608     0.690550  0.288942
     10.0     Michmoret-Hadera    2022-07-29       0  0.369196     0.656833  0.287637
      3.0      Kiryat Yam-Acco    2012-07-04       1  0.643721     0.359251  0.284469
      2.0       Acco-Nahariyah    2012-07-02       0  0.547448     0.264859  0.282590
     14.0     Tel Aviv-Herzlia    2012-06-03       1  0.618451     0.336455  0.281996
     18.0      Ashdod-Palmahim    2019-04-27       1  0.658926     0.377155  0.281770
     19.0      Ashkelon-Ashdod    2024-02-06       0  0.584329     0.303169  0.281160
      6.0           Atlit-Tira    2019-07-25       0  0.221416     0.502428  0.281012
     10.0     Michmoret-Hadera    2014-07-27       0  0.601091     0.322368  0.278723
     15.0       Jaffa-Tel Aviv    2021-09-27       0  0.563092     0.290186  0.272907
     17.0      Palmahim-Rishon    2022-09-19       0  0.653369     0.380756  0.272613
```

## Performance By Beach (Aggregated)

The table below aggregates performance and context per beach across the full test subset, so you can see where the model predicts well vs poorly by location.

```
 beach_id           beach_name  n_samples  actual_positive_rate  pred_positive_rate  accuracy  precision   recall       f1  tp  fp  tn  fn  avg_probability  avg_confidence_distance  avg_observation_count_last  avg_diameter_cm_last  avg_sting_last
       18      Ashdod-Palmahim        106              0.443396            0.386792  0.698113   0.682927 0.595745 0.636364  28  13  46  19         0.413139                 0.169618                    3.716981             14.095201        0.292453
       15       Jaffa-Tel Aviv        135              0.288889            0.259259  0.748148   0.571429 0.512821 0.540541  20  15  81  19         0.358452                 0.207162                    3.955555              8.909014        0.348148
       16         Rishon-Jaffa         26              0.346154            0.538462  0.576923   0.428571 0.666667 0.521739   6   8   9   3         0.494917                 0.147498                    3.307692             16.143543        0.346154
       19      Ashkelon-Ashdod        178              0.314607            0.247191  0.730337   0.590909 0.464286 0.520000  26  18 104  30         0.349728                 0.212661                    2.977528              6.499172        0.151685
       17      Palmahim-Rishon        175              0.377143            0.377143  0.611429   0.484848 0.484848 0.484848  32  34  75  34         0.415075                 0.174290                    4.354286             11.724663        0.222857
       14     Tel Aviv-Herzlia        162              0.339506            0.160494  0.709877   0.653846 0.309091 0.419753  17   9  98  38         0.310387                 0.229422                    3.419753             23.357769        0.141975
        9  Hadera-Jisr a zarqa        108              0.314815            0.175926  0.712963   0.578947 0.323529 0.415094  11   8  66  23         0.330153                 0.197772                    2.777778             14.156635        0.166667
       11    Natanya-Michmoret        129              0.224806            0.155039  0.775194   0.500000 0.344828 0.408163  10  10  90  19         0.322593                 0.200813                    3.333333              6.481930        0.217054
        7            Dor-Atlit        243              0.288066            0.123457  0.744856   0.633333 0.271429 0.380000  19  11 162  51         0.292912                 0.231364                    3.654321             14.794077        0.127572
       12        Gaash-Natanya        117              0.341880            0.341880  0.572650   0.375000 0.375000 0.375000  15  25  52  25         0.397611                 0.187723                    3.059829             10.225498        0.264957
       10     Michmoret-Hadera        177              0.384181            0.220339  0.621469   0.512821 0.294118 0.373832  20  19  90  48         0.359630                 0.195070                    3.389831             14.813637        0.209040
        8     Jisr a zarqa-Dor         69              0.333333            0.144928  0.666667   0.500000 0.217391 0.303030   5   5  41  18         0.300772                 0.225122                    3.217391             83.042274        0.144928
        3      Kiryat Yam-Acco        138              0.376812            0.123188  0.630435   0.529412 0.173077 0.260870   9   8  78  43         0.324095                 0.187279                    2.956522             10.850775        0.130435
        5        Tira-Shiqmona        183              0.311475            0.120219  0.677596   0.454545 0.175439 0.253165  10  12 114  47         0.286413                 0.235987                    2.918033              8.143469        0.174863
       13        Herzlia-Gaash        143              0.279720            0.167832  0.664336   0.333333 0.200000 0.250000   8  16  87  32         0.327211                 0.193332                    2.951049             26.170450        0.272727
        6           Atlit-Tira        117              0.256410            0.094017  0.735043   0.454545 0.166667 0.243902   5   6  81  25         0.295408                 0.215791                    2.410256              9.842592        0.153846
        1 Nahariya-Rosh Hniqra        144              0.326389            0.138889  0.645833   0.400000 0.170213 0.238806   8  12  85  39         0.340465                 0.181948                    4.250000              5.919527        0.194444
        2       Acco-Nahariyah         79              0.379747            0.088608  0.632911   0.571429 0.133333 0.216216   4   3  46  26         0.333490                 0.182093                    2.784810             12.747363        0.164557
        4  Shiqmona-Kiryat yam        105              0.304762            0.076190  0.695238   0.500000 0.125000 0.200000   4   4  69  28         0.284461                 0.226787                    2.800000              8.979285        0.047619
       20        Gaza-Ashkelon          1              0.000000            0.000000  1.000000   0.000000 0.000000 0.000000   0   0   1   0         0.293084                 0.206916                    2.000000             24.750000        0.000000
```

Beach-level CSV saved to: reports/qualitative_by_beach.csv

## Interpretation Notes

- The primary model misses more positives than negatives (higher FN), suggesting conservative bloom detection.
- TP/TN samples illustrate where the model is confident and likely exploiting stable seasonal/observation patterns.
- FP/FN samples highlight specific input conditions where the model underperforms and may require feature or threshold tuning.
