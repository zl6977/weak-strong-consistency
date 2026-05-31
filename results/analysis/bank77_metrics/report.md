# BANK77 Consistency Metrics

Dataset: BANKING77 test.

Role assignment: strong1 = Gemma31B, strong2 = Qwen27B, middle = Qwen9B, weak = Qwen35B.

## Single Model Accuracy
| role | run | accuracy | id_accuracy | oos_accuracy_recall | oos_precision | pred_oos_rate |
| --- | --- | --- | --- | --- | --- | --- |
| strong1 | gemma-4-31B-banking77-run0 | 0.8013 | 0.8013 |  |  | 0.0000 |
| strong2 | Qwen3.6-27B-banking77-run0 | 0.7792 | 0.7792 |  |  | 0.0000 |
| middle | Qwen3.5-9B-banking77-run0 | 0.5821 | 0.5821 |  |  | 0.0000 |
| weak | Qwen3.6-35B-A3B-banking77-run0 | 0.5669 | 0.5669 |  |  | 0.0000 |

## Single Model Accuracy by Split
| role | split | n | accuracy | pred_ood_rate | run |
| --- | --- | --- | --- | --- | --- |
| strong1 | overall | 3080 | 0.8013 | 0.0000 | gemma-4-31B-banking77-run0 |
| strong1 | id | 3080 | 0.8013 | 0.0000 | gemma-4-31B-banking77-run0 |
| strong1 | ood | 0 |  |  | gemma-4-31B-banking77-run0 |
| strong2 | overall | 3080 | 0.7792 | 0.0000 | Qwen3.6-27B-banking77-run0 |
| strong2 | id | 3080 | 0.7792 | 0.0000 | Qwen3.6-27B-banking77-run0 |
| strong2 | ood | 0 |  |  | Qwen3.6-27B-banking77-run0 |
| middle | overall | 3080 | 0.5821 | 0.0000 | Qwen3.5-9B-banking77-run0 |
| middle | id | 3080 | 0.5821 | 0.0000 | Qwen3.5-9B-banking77-run0 |
| middle | ood | 0 |  |  | Qwen3.5-9B-banking77-run0 |
| weak | overall | 3080 | 0.5669 | 0.0000 | Qwen3.6-35B-A3B-banking77-run0 |
| weak | id | 3080 | 0.5669 | 0.0000 | Qwen3.6-35B-A3B-banking77-run0 |
| weak | ood | 0 |  |  | Qwen3.6-35B-A3B-banking77-run0 |

## Agreement Reliability
| pair | coverage | acc_at_agree | acc_at_disagree | reliability_gap | error_capture_at_disagree | n_agree | n_disagree |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | 0.8825 | 0.8536 | 0.4088 | 0.4447 | 0.3497 | 2718 | 362 |
| strong1-middle | 0.6416 | 0.8639 | 0.6893 | 0.1746 | 0.5605 | 1976 | 1104 |
| strong1-weak | 0.6211 | 0.8714 | 0.6864 | 0.1850 | 0.5980 | 1913 | 1167 |

## ID/OOD Decomposition
| pair | split | n | coverage | acc_at_agree | acc_at_disagree | error_capture_at_disagree |
| --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | 3080 | 0.8825 | 0.8536 | 0.4088 | 0.3497 |
| strong1-strong2 | id | 3080 | 0.8825 | 0.8536 | 0.4088 | 0.3497 |
| strong1-strong2 | ood | 0 |  |  |  |  |
| strong1-middle | overall | 3080 | 0.6416 | 0.8639 | 0.6893 | 0.5605 |
| strong1-middle | id | 3080 | 0.6416 | 0.8639 | 0.6893 | 0.5605 |
| strong1-middle | ood | 0 |  |  |  |  |
| strong1-weak | overall | 3080 | 0.6211 | 0.8714 | 0.6864 | 0.5980 |
| strong1-weak | id | 3080 | 0.6211 | 0.8714 | 0.6864 | 0.5980 |
| strong1-weak | ood | 0 |  |  |  |  |

## Confidence x Agreement
| pair | split | confidence | agreement | n | coverage | accuracy | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | high | agree | 1512 | 0.4909 | 0.9312 | 0.0688 |
| strong1-strong2 | overall | high | disagree | 28 | 0.0091 | 0.7857 | 0.2143 |
| strong1-strong2 | overall | low | agree | 1206 | 0.3916 | 0.7562 | 0.2438 |
| strong1-strong2 | overall | low | disagree | 334 | 0.1084 | 0.3772 | 0.6228 |
| strong1-middle | overall | high | agree | 1142 | 0.3708 | 0.9431 | 0.0569 |
| strong1-middle | overall | high | disagree | 398 | 0.1292 | 0.8869 | 0.1131 |
| strong1-middle | overall | low | agree | 834 | 0.2708 | 0.7554 | 0.2446 |
| strong1-middle | overall | low | disagree | 706 | 0.2292 | 0.5779 | 0.4221 |
| strong1-weak | overall | high | agree | 1117 | 0.3627 | 0.9391 | 0.0609 |
| strong1-weak | overall | high | disagree | 423 | 0.1373 | 0.9007 | 0.0993 |
| strong1-weak | overall | low | agree | 796 | 0.2584 | 0.7764 | 0.2236 |
| strong1-weak | overall | low | disagree | 744 | 0.2416 | 0.5645 | 0.4355 |

## Confidence x Agreement by Split
| pair | split | confidence | agreement | n | coverage | accuracy | error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | high | agree | 1512 | 0.4909 | 0.9312 | 0.0688 |
| strong1-strong2 | overall | high | disagree | 28 | 0.0091 | 0.7857 | 0.2143 |
| strong1-strong2 | overall | low | agree | 1206 | 0.3916 | 0.7562 | 0.2438 |
| strong1-strong2 | overall | low | disagree | 334 | 0.1084 | 0.3772 | 0.6228 |
| strong1-strong2 | id | high | agree | 1512 | 0.4909 | 0.9312 | 0.0688 |
| strong1-strong2 | id | high | disagree | 28 | 0.0091 | 0.7857 | 0.2143 |
| strong1-strong2 | id | low | agree | 1206 | 0.3916 | 0.7562 | 0.2438 |
| strong1-strong2 | id | low | disagree | 334 | 0.1084 | 0.3772 | 0.6228 |
| strong1-strong2 | ood | high | agree | 0 |  |  |  |
| strong1-strong2 | ood | high | disagree | 0 |  |  |  |
| strong1-strong2 | ood | low | agree | 0 |  |  |  |
| strong1-strong2 | ood | low | disagree | 0 |  |  |  |
| strong1-middle | overall | high | agree | 1142 | 0.3708 | 0.9431 | 0.0569 |
| strong1-middle | overall | high | disagree | 398 | 0.1292 | 0.8869 | 0.1131 |
| strong1-middle | overall | low | agree | 834 | 0.2708 | 0.7554 | 0.2446 |
| strong1-middle | overall | low | disagree | 706 | 0.2292 | 0.5779 | 0.4221 |
| strong1-middle | id | high | agree | 1142 | 0.3708 | 0.9431 | 0.0569 |
| strong1-middle | id | high | disagree | 398 | 0.1292 | 0.8869 | 0.1131 |
| strong1-middle | id | low | agree | 834 | 0.2708 | 0.7554 | 0.2446 |
| strong1-middle | id | low | disagree | 706 | 0.2292 | 0.5779 | 0.4221 |
| strong1-middle | ood | high | agree | 0 |  |  |  |
| strong1-middle | ood | high | disagree | 0 |  |  |  |
| strong1-middle | ood | low | agree | 0 |  |  |  |
| strong1-middle | ood | low | disagree | 0 |  |  |  |
| strong1-weak | overall | high | agree | 1117 | 0.3627 | 0.9391 | 0.0609 |
| strong1-weak | overall | high | disagree | 423 | 0.1373 | 0.9007 | 0.0993 |
| strong1-weak | overall | low | agree | 796 | 0.2584 | 0.7764 | 0.2236 |
| strong1-weak | overall | low | disagree | 744 | 0.2416 | 0.5645 | 0.4355 |
| strong1-weak | id | high | agree | 1117 | 0.3627 | 0.9391 | 0.0609 |
| strong1-weak | id | high | disagree | 423 | 0.1373 | 0.9007 | 0.0993 |
| strong1-weak | id | low | agree | 796 | 0.2584 | 0.7764 | 0.2236 |
| strong1-weak | id | low | disagree | 744 | 0.2416 | 0.5645 | 0.4355 |
| strong1-weak | ood | high | agree | 0 |  |  |  |
| strong1-weak | ood | high | disagree | 0 |  |  |  |
| strong1-weak | ood | low | agree | 0 |  |  |  |
| strong1-weak | ood | low | disagree | 0 |  |  |  |

## Confidence Agreement Penalty
| pair | split | confidence | agree_accuracy | disagree_accuracy | accuracy_penalty | error_rate_increase | agree_n | disagree_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-middle | id | high | 0.9431 | 0.8869 | 0.0561 | 0.0561 | 1142 | 398 |
| strong1-middle | id | low | 0.7554 | 0.5779 | 0.1775 | 0.1775 | 834 | 706 |
| strong1-middle | ood | high |  |  |  |  | 0 | 0 |
| strong1-middle | ood | low |  |  |  |  | 0 | 0 |
| strong1-middle | overall | high | 0.9431 | 0.8869 | 0.0561 | 0.0561 | 1142 | 398 |
| strong1-middle | overall | low | 0.7554 | 0.5779 | 0.1775 | 0.1775 | 834 | 706 |
| strong1-strong2 | id | high | 0.9312 | 0.7857 | 0.1455 | 0.1455 | 1512 | 28 |
| strong1-strong2 | id | low | 0.7562 | 0.3772 | 0.3790 | 0.3790 | 1206 | 334 |
| strong1-strong2 | ood | high |  |  |  |  | 0 | 0 |
| strong1-strong2 | ood | low |  |  |  |  | 0 | 0 |
| strong1-strong2 | overall | high | 0.9312 | 0.7857 | 0.1455 | 0.1455 | 1512 | 28 |
| strong1-strong2 | overall | low | 0.7562 | 0.3772 | 0.3790 | 0.3790 | 1206 | 334 |
| strong1-weak | id | high | 0.9391 | 0.9007 | 0.0384 | 0.0384 | 1117 | 423 |
| strong1-weak | id | low | 0.7764 | 0.5645 | 0.2119 | 0.2119 | 796 | 744 |
| strong1-weak | ood | high |  |  |  |  | 0 | 0 |
| strong1-weak | ood | low |  |  |  |  | 0 | 0 |
| strong1-weak | overall | high | 0.9391 | 0.9007 | 0.0384 | 0.0384 | 1117 | 423 |
| strong1-weak | overall | low | 0.7764 | 0.5645 | 0.2119 | 0.2119 | 796 | 744 |

## Correctness Ranking Scores
| pair | score | auroc_correctness | auprc_correctness |
| --- | --- | --- | --- |
| strong1-strong2 | agreement_binary | 0.6449 | 0.8504 |
| strong1-strong2 | strong1_id_conf | 0.8001 | 0.9254 |
| strong1-strong2 | 0.5_conf_plus_0.5_agree | 0.8043 | 0.9270 |
| strong1-strong2 | 1-normalized_js | 0.8085 | 0.9435 |
| strong1-strong2 | cosine_similarity | 0.8201 | 0.9459 |
| strong1-strong2 | top3_overlap | 0.2901 | 0.7538 |
| strong1-middle | agreement_binary | 0.6261 | 0.8446 |
| strong1-middle | strong1_id_conf | 0.8001 | 0.9254 |
| strong1-middle | 0.5_conf_plus_0.5_agree | 0.7604 | 0.9195 |
| strong1-middle | 1-normalized_js | 0.6700 | 0.8920 |
| strong1-middle | cosine_similarity | 0.6773 | 0.8952 |
| strong1-middle | top3_overlap | 0.4189 | 0.7827 |
| strong1-weak | agreement_binary | 0.6367 | 0.8487 |
| strong1-weak | strong1_id_conf | 0.8001 | 0.9254 |
| strong1-weak | 0.5_conf_plus_0.5_agree | 0.7645 | 0.9189 |
| strong1-weak | 1-normalized_js | 0.6784 | 0.8869 |
| strong1-weak | cosine_similarity | 0.6848 | 0.8887 |
| strong1-weak | top3_overlap | 0.4200 | 0.7836 |

## MSP Baseline (Selective Prediction)
For each coverage point used by an agreement gate, MSP reports the accepted accuracy of a confidence-only gate.

| method | coverage | accepted_accuracy | risk | n_accepted |
| --- | --- | --- | --- | --- |
| msp | 0.6211 | 0.9195 | 0.0805 | 1913 |
| msp | 0.6416 | 0.9185 | 0.0815 | 1976 |
| msp | 0.8825 | 0.8547 | 0.1453 | 2718 |

## Agreement vs MSP Comparison
| pair | coverage | acc_at_agree | msp_acc | delta |
| --- | --- | --- | --- | --- |
| strong1-strong2 | 0.8825 | 0.8536 | 0.8547 | -0.0011 |
| strong1-middle | 0.6416 | 0.8639 | 0.9185 | -0.0547 |
| strong1-weak | 0.6211 | 0.8714 | 0.9195 | -0.0481 |
