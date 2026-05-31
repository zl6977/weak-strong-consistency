# CLINC Paper Metrics

Dataset: CLINC plus test.

Role assignment: strong1 = Gemma31B, strong2 = Qwen27B, middle = Qwen9B, weak = Qwen35B.

## Single Model Accuracy
| role | run | accuracy | id_accuracy | oos_accuracy_recall | oos_precision | pred_oos_rate |
| --- | --- | --- | --- | --- | --- | --- |
| strong1 | gemma-4-31B-run0-all | 0.8615 | 0.9184 | 0.6050 | 0.9409 | 0.1169 |
| strong2 | Qwen3.6-27B-run0-all | 0.7384 | 0.8929 | 0.0430 | 0.8958 | 0.0087 |
| middle | qwen3.5-9b-clinc-run0-all | 0.6373 | 0.7756 | 0.0150 | 0.6250 | 0.0044 |
| weak | Qwen3.6-35B-run0-all | 0.5838 | 0.7104 | 0.0140 | 0.5833 | 0.0044 |

## Agreement-Conditioned Accuracy by Split
| pair | split | n | coverage | acc_at_agree | acc_at_disagree | reliability_gap | error_capture_at_disagree | n_agree | n_disagree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong1-strong2 | overall | 5500 | 0.7767 | 0.9228 | 0.6482 | 0.2745 | 0.5669 | 4272 | 1228 |
| strong1-strong2 | id | 4500 | 0.9109 | 0.9541 | 0.5536 | 0.4005 | 0.4877 | 4099 | 401 |
| strong1-strong2 | ood | 1000 | 0.1730 | 0.1792 | 0.6941 | -0.5149 | 0.6405 | 173 | 827 |
| strong1-middle | overall | 5500 | 0.6745 | 0.9197 | 0.7408 | 0.1789 | 0.6089 | 3710 | 1790 |
| strong1-middle | id | 4500 | 0.7911 | 0.9545 | 0.7819 | 0.1726 | 0.5586 | 3560 | 940 |
| strong1-middle | ood | 1000 | 0.1500 | 0.0933 | 0.6953 | -0.6020 | 0.6557 | 150 | 850 |
| strong1-weak | overall | 5500 | 0.6115 | 0.9260 | 0.7599 | 0.1660 | 0.6732 | 3363 | 2137 |
| strong1-weak | id | 4500 | 0.7173 | 0.9607 | 0.8113 | 0.1493 | 0.6540 | 3228 | 1272 |
| strong1-weak | ood | 1000 | 0.1350 | 0.0963 | 0.6844 | -0.5881 | 0.6911 | 135 | 865 |
