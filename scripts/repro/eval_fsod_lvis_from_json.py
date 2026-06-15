import copy
import json
import logging
from pathlib import Path

from fsdet.evaluation.lvis_evaluation import _evaluate_predictions_on_lvis
from lvis import LVIS

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

ANN_PATH = "datasets/lvis/lvis_v0.5_val.json"
RESULTS_PATH = "outputs/repro/fsod_full/MSCOSINE_all_fc/vis_text/inference/lvis_instances_results.json"
OUTPUT_PATH = "outputs/repro/fsod_full/MSCOSINE_all_fc/vis_text/inference/lvis_eval_results_from_json.json"


def main():
    print(f"ann={ANN_PATH}")
    print(f"results={RESULTS_PATH}")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    print(f"num_results={len(results)}")
    print(f"keys={sorted(results[0].keys()) if results else []}")

    lvis_gt = LVIS(ANN_PATH)
    summary = {"bbox": _evaluate_predictions_on_lvis(lvis_gt, results, "bbox")}
    if results and "segmentation" in results[0]:
        segm_results = []
        for result in results:
            item = copy.deepcopy(result)
            item.pop("bbox", None)
            segm_results.append(item)
        summary["segm"] = _evaluate_predictions_on_lvis(lvis_gt, segm_results, "segm")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"saved={OUTPUT_PATH}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
