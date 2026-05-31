# BANK77 Paper Metrics

Dataset: BANKING77 test.

Role assignment: strong1 = Gemma31B, strong2 = Qwen27B, middle = Qwen9B, weak = Qwen35B.

## Single Model Accuracy
| role | run | accuracy | id_accuracy | oos_accuracy_recall | oos_precision | pred_oos_rate |
| --- | --- | --- | --- | --- | --- | --- |
| strong1 | gemma-4-31B-banking77-run0 | 0.8013 | 0.8013 |  |  | 0.0000 |
| strong2 | Qwen3.6-27B-banking77-run0 | 0.7792 | 0.7792 |  |  | 0.0000 |
| middle | Qwen3.5-9B-banking77-run0 | 0.5821 | 0.5821 |  |  | 0.0000 |
| weak | Qwen3.6-35B-A3B-banking77-run0 | 0.5669 | 0.5669 |  |  | 0.0000 |

## Agreement-Conditioned Accuracy by Split
| pair | split | n | coverage | acc_at_agree | acc_at_disagree | reliability_gap | error_capture_at_disagree | n_agree | n_disagree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | 3080 | 0.8825 | 0.8536 | 0.4088 | 0.4447 | 0.3497 | 2718 | 362 |
| strong1-strong2 | id | 3080 | 0.8825 | 0.8536 | 0.4088 | 0.4447 | 0.3497 | 2718 | 362 |
| strong1-strong2 | ood | 0 |  |  |  |  |  | 0 | 0 |
| strong1-middle | overall | 3080 | 0.6416 | 0.8639 | 0.6893 | 0.1746 | 0.5605 | 1976 | 1104 |
| strong1-middle | id | 3080 | 0.6416 | 0.8639 | 0.6893 | 0.1746 | 0.5605 | 1976 | 1104 |
| strong1-middle | ood | 0 |  |  |  |  |  | 0 | 0 |
| strong1-weak | overall | 3080 | 0.6211 | 0.8714 | 0.6864 | 0.1850 | 0.5980 | 1913 | 1167 |
| strong1-weak | id | 3080 | 0.6211 | 0.8714 | 0.6864 | 0.1850 | 0.5980 | 1913 | 1167 |
| strong1-weak | ood | 0 |  |  |  |  |  | 0 | 0 |

## BANKING77 High-Confidence Complementarity
| pair | agree_accuracy | disagree_accuracy | agree_n | disagree_n |
| --- | --- | --- | --- | --- |
| strong1-strong2 | 0.9263 | 0.7879 | 1507 | 33 |
| strong1-middle | 0.9381 | 0.8829 | 1130 | 410 |
| strong1-weak | 0.9341 | 0.8958 | 1108 | 432 |

## BANKING77 MSP Baseline
| method | coverage | accepted_accuracy | risk | n_accepted |
| --- | --- | --- | --- | --- |
| msp | 0.6211 | 0.9195 | 0.0805 | 1913 |
| msp | 0.6416 | 0.9185 | 0.0815 | 1976 |
| msp | 0.8825 | 0.8547 | 0.1453 | 2718 |
