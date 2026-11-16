import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_model_metadata_verification():
    metadata_file = PROJECT_ROOT / "models" / "model_metadata.json"
    assert metadata_file.exists(), "models/model_metadata.json must exist"

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Output classes order check
    assert meta["classes"] == ["Benign", "Malignant", "Normal"], "Classes must be Benign, Malignant, Normal"

    # 2. Input shape
    assert meta["input_shape"] == [224, 224, 3]

    # 3. Base model architecture
    assert "EfficientNetB0" in meta["architecture"]["base_model"]
    assert meta["architecture"]["output_layer"] == "Dense(3, activation='softmax', name='predictions')"

    # 4. Dataset counts check
    split = meta["dataset_split"]
    assert split["test_counts"]["total"] == 166
    assert split["test_counts"]["Benign"] == 18
    assert split["test_counts"]["Malignant"] == 85
    assert split["test_counts"]["Normal"] == 63

    # 5. Verified evaluation of original model check
    orig = meta["verified_evaluation"]["original_best_model"]
    assert orig["test_accuracy"] == 0.9458
    assert orig["test_loss"] == 0.2184
    assert orig["confusion_matrix"] == [
        [16, 0, 2],
        [1, 83, 1],
        [3, 2, 58]
    ]

    # Check matrix sum equals test image count
    cm_sum = sum(sum(row) for row in orig["confusion_matrix"])
    assert cm_sum == 166

    # 6. Check fine tuned model is not conflated with original model
    ft = meta["verified_evaluation"]["fine_tuned_experiment"]
    assert ft["test_accuracy"] == 0.6867
    assert ft["test_loss"] == 0.6528
