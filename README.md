# WiFi-CSI ReID

**Person re-identification using Wi-Fi Channel State Information (CSI).**

WiFi-CSI ReID is an experimental machine-learning project investigating whether Wi-Fi signal measurements can distinguish known individuals across separate recording sessions. The project evaluates identity retrieval from wireless sensing data, without relying on conventional camera images.

## Results at a glance

| Evaluation result | Value |
| --- | ---: |
| Reported system · Rank-1 accuracy | **90.3%** |
| Reported system · Rank-5 accuracy | **97.7%** |
| Reported system · mean Average Precision (mAP) | **0.862** |
| Best observed configuration · Rank-1 accuracy | **90.7%** |
| Controlled reference · Rank-1 accuracy | **79.1%** |

These results demonstrate the potential of Wi-Fi CSI as a signal source for **closed-set person re-identification** in a controlled experimental setting.

## Project showcase

The repository includes an interactive presentation of the research problem, experimental observations, and reported outcomes in [`website/index.html`](website/index.html).

## Evaluation scope and limitations

The reported results use the [Wi-PER81](https://doi.org/10.6084/m9.figshare.26984497) benchmark, covering 81 identities. The evaluation is **closed-set**: the tested identities are represented in the training data, while evaluation uses a different acquisition session. The findings do not establish identification of previously unseen individuals or robustness across unrelated environments, devices, or long time intervals.

The dataset is not redistributed here. Research implementation details, training procedures, and experiment-level materials are intentionally omitted from this public-facing summary.

## Dataset reference

Cascio, M., Cinque, L., Distante, D., Foresti, G. L., & Fagioli, A. *A Benchmark Dataset for Radio Signal Image-based Person Re-Identification.* **Scientific Data**, 12, 1522 (2025). https://doi.org/10.1038/s41597-025-05804-0

---

<sub>The original experimentation was conducted collaboratively with additional team members.</sub>
