# CLINC Consistency Metrics

Dataset: CLINC plus test.

Role assignment: strong1 = Gemma31B, strong2 = Qwen27B, middle = Qwen9B, weak = Qwen35B.

## Single Model Accuracy
| role | run | accuracy | id_accuracy | oos_accuracy_recall | oos_precision | pred_oos_rate |
| --- | --- | --- | --- | --- | --- | --- |
| strong1 | gemma-4-31B-run0-all | 0.8615 | 0.9184 | 0.6050 | 0.9409 | 0.1169 |
| strong2 | Qwen3.6-27B-run0-all | 0.7384 | 0.8929 | 0.0430 | 0.8958 | 0.0087 |
| middle | qwen3.5-9b-clinc-run0-all | 0.6373 | 0.7756 | 0.0150 | 0.6250 | 0.0044 |
| weak | Qwen3.6-35B-run0-all | 0.5838 | 0.7104 | 0.0140 | 0.5833 | 0.0044 |

## Single Model Accuracy by Split
| role | split | n | accuracy | pred_ood_rate | run |
| --- | --- | --- | --- | --- | --- |
| strong1 | overall | 5500 | 0.8615 | 0.1169 | gemma-4-31B-run0-all |
| strong1 | id | 4500 | 0.9184 | 0.0084 | gemma-4-31B-run0-all |
| strong1 | ood | 1000 | 0.6050 | 0.6050 | gemma-4-31B-run0-all |
| strong2 | overall | 5500 | 0.7384 | 0.0087 | Qwen3.6-27B-run0-all |
| strong2 | id | 4500 | 0.8929 | 0.0011 | Qwen3.6-27B-run0-all |
| strong2 | ood | 1000 | 0.0430 | 0.0430 | Qwen3.6-27B-run0-all |
| middle | overall | 5500 | 0.6373 | 0.0044 | qwen3.5-9b-clinc-run0-all |
| middle | id | 4500 | 0.7756 | 0.0020 | qwen3.5-9b-clinc-run0-all |
| middle | ood | 1000 | 0.0150 | 0.0150 | qwen3.5-9b-clinc-run0-all |
| weak | overall | 5500 | 0.5838 | 0.0044 | Qwen3.6-35B-run0-all |
| weak | id | 4500 | 0.7104 | 0.0022 | Qwen3.6-35B-run0-all |
| weak | ood | 1000 | 0.0140 | 0.0140 | Qwen3.6-35B-run0-all |

## Agreement Reliability
| pair | coverage | acc_at_agree | acc_at_disagree | reliability_gap | error_capture_at_disagree | n_agree | n_disagree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | 0.7767 | 0.9228 | 0.6482 | 0.2745 | 0.5669 | 4272 | 1228 |
| strong1-middle | 0.6745 | 0.9197 | 0.7408 | 0.1789 | 0.6089 | 3710 | 1790 |
| strong1-weak | 0.6115 | 0.9260 | 0.7599 | 0.1660 | 0.6732 | 3363 | 2137 |

## ID/OOD Decomposition
| pair | split | n | coverage | acc_at_agree | acc_at_disagree | error_capture_at_disagree |
| --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | 5500 | 0.7767 | 0.9228 | 0.6482 | 0.5669 |
| strong1-strong2 | id | 4500 | 0.9109 | 0.9541 | 0.5536 | 0.4877 |
| strong1-strong2 | ood | 1000 | 0.1730 | 0.1792 | 0.6941 | 0.6405 |
| strong1-middle | overall | 5500 | 0.6745 | 0.9197 | 0.7408 | 0.6089 |
| strong1-middle | id | 4500 | 0.7911 | 0.9545 | 0.7819 | 0.5586 |
| strong1-middle | ood | 1000 | 0.1500 | 0.0933 | 0.6953 | 0.6557 |
| strong1-weak | overall | 5500 | 0.6115 | 0.9260 | 0.7599 | 0.6732 |
| strong1-weak | id | 4500 | 0.7173 | 0.9607 | 0.8113 | 0.6540 |
| strong1-weak | ood | 1000 | 0.1350 | 0.0963 | 0.6844 | 0.6911 |

## Confidence x Agreement
| pair | split | confidence | agreement | n | coverage | accuracy | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | high | agree | 2715 | 0.4936 | 0.9646 | 0.0354 |
| strong1-strong2 | overall | high | disagree | 35 | 0.0064 | 0.8571 | 0.1429 |
| strong1-strong2 | overall | low | agree | 1557 | 0.2831 | 0.8497 | 0.1503 |
| strong1-strong2 | overall | low | disagree | 1193 | 0.2169 | 0.6421 | 0.3579 |
| strong1-middle | overall | high | agree | 2415 | 0.4391 | 0.9669 | 0.0331 |
| strong1-middle | overall | high | disagree | 335 | 0.0609 | 0.9373 | 0.0627 |
| strong1-middle | overall | low | agree | 1295 | 0.2355 | 0.8317 | 0.1683 |
| strong1-middle | overall | low | disagree | 1455 | 0.2645 | 0.6955 | 0.3045 |
| strong1-weak | overall | high | agree | 2190 | 0.3982 | 0.9721 | 0.0279 |
| strong1-weak | overall | high | disagree | 560 | 0.1018 | 0.9286 | 0.0714 |
| strong1-weak | overall | low | agree | 1173 | 0.2133 | 0.8397 | 0.1603 |
| strong1-weak | overall | low | disagree | 1577 | 0.2867 | 0.7001 | 0.2999 |

## Confidence x Agreement by Split
| pair | split | confidence | agreement | n | coverage | accuracy | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | high | agree | 2715 | 0.4936 | 0.9646 | 0.0354 |
| strong1-strong2 | overall | high | disagree | 35 | 0.0064 | 0.8571 | 0.1429 |
| strong1-strong2 | overall | low | agree | 1557 | 0.2831 | 0.8497 | 0.1503 |
| strong1-strong2 | overall | low | disagree | 1193 | 0.2169 | 0.6421 | 0.3579 |
| strong1-strong2 | id | high | agree | 2684 | 0.5964 | 0.9758 | 0.0242 |
| strong1-strong2 | id | high | disagree | 33 | 0.0073 | 0.9091 | 0.0909 |
| strong1-strong2 | id | low | agree | 1415 | 0.3144 | 0.9131 | 0.0869 |
| strong1-strong2 | id | low | disagree | 368 | 0.0818 | 0.5217 | 0.4783 |
| strong1-strong2 | ood | high | agree | 31 | 0.0310 | 0.0000 | 1.0000 |
| strong1-strong2 | ood | high | disagree | 2 | 0.0020 | 0.0000 | 1.0000 |
| strong1-strong2 | ood | low | agree | 142 | 0.1420 | 0.2183 | 0.7817 |
| strong1-strong2 | ood | low | disagree | 825 | 0.8250 | 0.6958 | 0.3042 |
| strong1-middle | overall | high | agree | 2415 | 0.4391 | 0.9669 | 0.0331 |
| strong1-middle | overall | high | disagree | 335 | 0.0609 | 0.9373 | 0.0627 |
| strong1-middle | overall | low | agree | 1295 | 0.2355 | 0.8317 | 0.1683 |
| strong1-middle | overall | low | disagree | 1455 | 0.2645 | 0.6955 | 0.3045 |
| strong1-middle | id | high | agree | 2389 | 0.5309 | 0.9774 | 0.0226 |
| strong1-middle | id | high | disagree | 328 | 0.0729 | 0.9573 | 0.0427 |
| strong1-middle | id | low | agree | 1171 | 0.2602 | 0.9078 | 0.0922 |
| strong1-middle | id | low | disagree | 612 | 0.1360 | 0.6879 | 0.3121 |
| strong1-middle | ood | high | agree | 26 | 0.0260 | 0.0000 | 1.0000 |
| strong1-middle | ood | high | disagree | 7 | 0.0070 | 0.0000 | 1.0000 |
| strong1-middle | ood | low | agree | 124 | 0.1240 | 0.1129 | 0.8871 |
| strong1-middle | ood | low | disagree | 843 | 0.8430 | 0.7011 | 0.2989 |
| strong1-weak | overall | high | agree | 2190 | 0.3982 | 0.9721 | 0.0279 |
| strong1-weak | overall | high | disagree | 560 | 0.1018 | 0.9286 | 0.0714 |
| strong1-weak | overall | low | agree | 1173 | 0.2133 | 0.8397 | 0.1603 |
| strong1-weak | overall | low | disagree | 1577 | 0.2867 | 0.7001 | 0.2999 |
| strong1-weak | id | high | agree | 2169 | 0.4820 | 0.9816 | 0.0184 |
| strong1-weak | id | high | disagree | 548 | 0.1218 | 0.9489 | 0.0511 |
| strong1-weak | id | low | agree | 1059 | 0.2353 | 0.9178 | 0.0822 |
| strong1-weak | id | low | disagree | 724 | 0.1609 | 0.7072 | 0.2928 |
| strong1-weak | ood | high | agree | 21 | 0.0210 | 0.0000 | 1.0000 |
| strong1-weak | ood | high | disagree | 12 | 0.0120 | 0.0000 | 1.0000 |
| strong1-weak | ood | low | agree | 114 | 0.1140 | 0.1140 | 0.8860 |
| strong1-weak | ood | low | disagree | 853 | 0.8530 | 0.6940 | 0.3060 |

## Confidence Agreement Penalty
| pair | split | confidence | agree_accuracy | disagree_accuracy | accuracy_penalty | error_rate_increase | agree_n | disagree_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-middle | id | high | 0.9774 | 0.9573 | 0.0201 | 0.0201 | 2389 | 328 |
| strong1-middle | id | low | 0.9078 | 0.6879 | 0.2199 | 0.2199 | 1171 | 612 |
| strong1-middle | ood | high | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 26 | 7 |
| strong1-middle | ood | low | 0.1129 | 0.7011 | -0.5882 | -0.5882 | 124 | 843 |
| strong1-middle | overall | high | 0.9669 | 0.9373 | 0.0296 | 0.0296 | 2415 | 335 |
| strong1-middle | overall | low | 0.8317 | 0.6955 | 0.1361 | 0.1361 | 1295 | 1455 |
| strong1-strong2 | id | high | 0.9758 | 0.9091 | 0.0667 | 0.0667 | 2684 | 33 |
| strong1-strong2 | id | low | 0.9131 | 0.5217 | 0.3913 | 0.3913 | 1415 | 368 |
| strong1-strong2 | ood | high | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 31 | 2 |
| strong1-strong2 | ood | low | 0.2183 | 0.6958 | -0.4774 | -0.4774 | 142 | 825 |
| strong1-strong2 | overall | high | 0.9646 | 0.8571 | 0.1075 | 0.1075 | 2715 | 35 |
| strong1-strong2 | overall | low | 0.8497 | 0.6421 | 0.2076 | 0.2076 | 1557 | 1193 |
| strong1-weak | id | high | 0.9816 | 0.9489 | 0.0327 | 0.0327 | 2169 | 548 |
| strong1-weak | id | low | 0.9178 | 0.7072 | 0.2107 | 0.2107 | 1059 | 724 |
| strong1-weak | ood | high | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 21 | 12 |
| strong1-weak | ood | low | 0.1140 | 0.6940 | -0.5800 | -0.5800 | 114 | 853 |
| strong1-weak | overall | high | 0.9721 | 0.9286 | 0.0436 | 0.0436 | 2190 | 560 |
| strong1-weak | overall | low | 0.8397 | 0.7001 | 0.1397 | 0.1397 | 1173 | 1577 |

## Correctness Ranking Scores
| pair | score | auroc_correctness | auprc_correctness |
| --- | --- | --- | --- |
| strong1-strong2 | agreement_binary | 0.6995 | 0.9125 |
| strong1-strong2 | strong1_id_conf | 0.7687 | 0.9506 |
| strong1-strong2 | 0.5_conf_plus_0.5_agree | 0.7743 | 0.9515 |
| strong1-strong2 | 1-normalized_js | 0.7898 | 0.9610 |
| strong1-strong2 | cosine_similarity | 0.7988 | 0.9623 |
| strong1-strong2 | top3_overlap | 0.4285 | 0.8455 |
| strong1-middle | agreement_binary | 0.6645 | 0.9034 |
| strong1-middle | strong1_id_conf | 0.7687 | 0.9506 |
| strong1-middle | 0.5_conf_plus_0.5_agree | 0.7500 | 0.9475 |
| strong1-middle | 1-normalized_js | 0.7144 | 0.9386 |
| strong1-middle | cosine_similarity | 0.7178 | 0.9391 |
| strong1-middle | top3_overlap | 0.5045 | 0.8628 |
| strong1-weak | agreement_binary | 0.6652 | 0.9038 |
| strong1-weak | strong1_id_conf | 0.7687 | 0.9506 |
| strong1-weak | 0.5_conf_plus_0.5_agree | 0.7489 | 0.9475 |
| strong1-weak | 1-normalized_js | 0.7116 | 0.9389 |
| strong1-weak | cosine_similarity | 0.7143 | 0.9405 |
| strong1-weak | top3_overlap | 0.5029 | 0.8626 |

## MSP Baseline (Selective Prediction)
For each coverage point used by an agreement gate, MSP reports the accepted accuracy of a confidence-only gate.

| method | coverage | accepted_accuracy | risk | n_accepted |
| --- | --- | --- | --- | --- |
| msp | 0.6115 | 0.9566 | 0.0434 | 3363 |
| msp | 0.6745 | 0.9555 | 0.0445 | 3710 |
| msp | 0.7767 | 0.9434 | 0.0566 | 4272 |

## Agreement vs MSP Comparison
| pair | coverage | acc_at_agree | msp_acc | delta |
| --- | --- | --- | --- | --- |
| strong1-strong2 | 0.7767 | 0.9228 | 0.9434 | -0.0206 |
| strong1-middle | 0.6745 | 0.9197 | 0.9555 | -0.0358 |
| strong1-weak | 0.6115 | 0.9260 | 0.9566 | -0.0306 |
