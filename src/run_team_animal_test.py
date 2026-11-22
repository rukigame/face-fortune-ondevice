import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ai_edge_litert.interpreter import Interpreter


# ==================================================
# Paths
# ==================================================

ROOT = Path(__file__).resolve().parent

ANIMAL_DIR = (
    ROOT
    / "animal_pokemon_final"
)

MODEL_PATH = (
    ANIMAL_DIR
    / "animal15_feature_extractor_finetuned.tflite"
)

PROTOTYPE_PATH = (
    ANIMAL_DIR
    / "animal15_multi_prototypes_finetuned.npz"
)

CNN_BASELINE_PATH = (
    ANIMAL_DIR
    / "animal15_cnn_baseline_finetuned.npz"
)

GEOMETRY_PROFILE_PATH = (
    ANIMAL_DIR
    / "animal15_geometry_profiles.npz"
)

HUMAN_GEOMETRY_PATH = (
    ANIMAL_DIR
    / "human_geometry_baseline.npz"
)

WFLW_GEOMETRY_AB_PATH = (
    ROOT
    / "human_geometry_baseline_wflw_provisional.npz"
)

POKEMON_CONFIG_PATH = (
    ANIMAL_DIR
    / "pokemon_matching_config.json"
)


# ==================================================
# Import existing Pokemon functions
# ==================================================

if str(ANIMAL_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ANIMAL_DIR)
    )


from test_pokemon_final import (
    get_style_scores,
    select_pokemon,
)


# ==================================================
# TFLite feature extractor
# ==================================================

interpreter = Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)


# ==================================================
# Load CNN prototype data
# ==================================================

prototype_data = np.load(
    str(PROTOTYPE_PATH)
)

multi_prototypes = {
    key: prototype_data[key]
    for key in prototype_data.files
}


# ==================================================
# Load CNN baseline
# ==================================================

cnn_data = np.load(
    str(CNN_BASELINE_PATH)
)

class_names = [
    str(x)
    for x in cnn_data["class_names"]
]

cnn_mean = {
    name: float(value)
    for name, value in zip(
        class_names,
        cnn_data["mean"]
    )
}

cnn_std = {
    name: float(value)
    for name, value in zip(
        class_names,
        cnn_data["std"]
    )
}


# ==================================================
# Load animal geometry profiles
# ==================================================

geometry_data = np.load(
    str(GEOMETRY_PROFILE_PATH)
)

animal_geometry_profiles = {
    key: geometry_data[key]
    for key in geometry_data.files
}


# ==================================================
# Load human geometry baseline
# ==================================================

human_geometry = np.load(
    str(HUMAN_GEOMETRY_PATH)
)

feature_names = [
    str(x)
    for x in human_geometry[
        "feature_names"
    ]
]

geometry_mean = {
    name: float(value)
    for name, value in zip(
        feature_names,
        human_geometry["mean"]
    )
}

geometry_std = {
    name: float(value)
    for name, value in zip(
        feature_names,
        human_geometry["std"]
    )
}

wflw_geometry_ab = None
wflw_geometry_ab_error = None
try:
    wflw_data = np.load(str(WFLW_GEOMETRY_AB_PATH), allow_pickle=False)
    wflw_feature_names = [str(value) for value in wflw_data["feature_names"]]
    if wflw_feature_names != feature_names:
        raise ValueError(
            "WFLW A/B baseline feature names/order mismatch: "
            f"expected {feature_names}, got {wflw_feature_names}"
        )
    wflw_geometry_ab = {
        "path": str(WFLW_GEOMETRY_AB_PATH),
        "feature_names": wflw_feature_names,
        "mean": {
            name: float(value)
            for name, value in zip(wflw_feature_names, wflw_data["mean"])
        },
        "std": {
            name: float(value)
            for name, value in zip(wflw_feature_names, wflw_data["std"])
        },
        "subject_count": int(wflw_data["subject_count"]),
        "capture_count": int(wflw_data["capture_count"]),
        "method": str(wflw_data["method"]),
        "created_at_utc": str(wflw_data["created_at_utc"]),
        "status": str(wflw_data["status"]),
    }
except (OSError, KeyError, TypeError, ValueError) as error:
    wflw_geometry_ab_error = str(error)


# ==================================================
# Utility
# ==================================================

def normalize_scores(scores):

    values = np.array(
        list(scores.values()),
        dtype=np.float32
    )

    minimum = float(
        values.min()
    )

    maximum = float(
        values.max()
    )

    return {
        key: float(
            (
                value
                -
                minimum
            )
            /
            (
                maximum
                -
                minimum
                +
                1e-8
            )
        )
        for key, value
        in scores.items()
    }


# ==================================================
# CNN feature extraction
# ==================================================

def extract_cnn_feature(
    image_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Cannot read image: "
            f"{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(
        np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    feature = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    feature = np.asarray(
        feature,
        dtype=np.float32
    )

    feature = (
        feature
        /
        (
            np.linalg.norm(feature)
            +
            1e-8
        )
    )

    return feature


# ==================================================
# CNN animal scores
# ==================================================

def get_cnn_scores_from_feature(feature):
    scores = {}
    diagnostics = {}

    for (
        animal,
        prototypes
    ) in multi_prototypes.items():

        similarities = []

        for prototype in prototypes:

            prototype = np.asarray(
                prototype,
                dtype=np.float32
            )

            prototype = (
                prototype
                /
                (
                    np.linalg.norm(
                        prototype
                    )
                    +
                    1e-8
                )
            )

            similarity = float(
                np.dot(
                    feature,
                    prototype
                )
            )

            similarities.append(
                similarity
            )

        raw_score = max(
            similarities
        )

        z_score_unclipped = (
            raw_score
            -
            cnn_mean[animal]
        ) / (
            cnn_std[animal]
            +
            1e-8
        )

        z_score_clipped = float(
            np.clip(
                z_score_unclipped,
                -3.0,
                3.0
            )
        )

        scores[animal] = z_score_clipped
        diagnostics[animal] = {
            "prototype_count": int(len(similarities)),
            "prototype_max_cosine": float(raw_score),
            "prototype_mean_cosine": float(np.mean(similarities)),
            "prototype_std_cosine": float(np.std(similarities)),
            "class_baseline_mean": float(cnn_mean[animal]),
            "class_baseline_std": float(cnn_std[animal]),
            "standardized_z_unclipped": float(z_score_unclipped),
            "standardized_z_clipped": z_score_clipped,
        }

    return scores, diagnostics


def get_cnn_scores(
    image_path
):

    feature = extract_cnn_feature(
        image_path
    )

    return get_cnn_scores_from_feature(feature)


def get_cnn_scores_multi(
    image_paths
):

    image_paths = [Path(path) for path in image_paths]
    if len(image_paths) < 2:
        raise ValueError("Multi-frame CNN requires at least two images")

    features = np.stack(
        [extract_cnn_feature(path) for path in image_paths],
        axis=0,
    ).astype(np.float32)

    aggregate_feature = np.mean(features, axis=0)
    aggregate_feature = aggregate_feature / (
        np.linalg.norm(aggregate_feature) + 1e-8
    )

    scores, diagnostics = get_cnn_scores_from_feature(aggregate_feature)
    per_frame_scores = [get_cnn_scores_from_feature(feature)[0] for feature in features]
    for animal in diagnostics:
        values = np.array(
            [frame_scores[animal] for frame_scores in per_frame_scores],
            dtype=np.float32,
        )
        diagnostics[animal].update(
            per_frame_standardized_z_mean=float(np.mean(values)),
            per_frame_standardized_z_std=float(np.std(values)),
            per_frame_standardized_z_min=float(np.min(values)),
            per_frame_standardized_z_max=float(np.max(values)),
        )
    frame_cosines = features @ aggregate_feature
    aggregation = {
        "frame_count": int(len(image_paths)),
        "image_names": [path.name for path in image_paths],
        "method": "mean of per-frame L2-normalized CNN embeddings, then L2 normalize",
        "mean_frame_to_aggregate_cosine": float(np.mean(frame_cosines)),
        "min_frame_to_aggregate_cosine": float(np.min(frame_cosines)),
        "max_frame_to_aggregate_cosine": float(np.max(frame_cosines)),
        "std_frame_to_aggregate_cosine": float(np.std(frame_cosines)),
    }
    return scores, diagnostics, aggregation


# ==================================================
# Convert team feature_values
# ==================================================

def team_features_to_geometry(
    feature_values
):

    required = [
        "face_hw_ratio",
        "right_eye_width_ratio",
        "left_eye_width_ratio",
        "eye_distance_ratio",
        "mouth_width_ratio",
        "jaw_width_ratio",
    ]

    for key in required:

        if key not in feature_values:

            raise KeyError(
                f"Missing team feature: "
                f"{key}"
            )

    face_hw_ratio = float(
        feature_values[
            "face_hw_ratio"
        ]
    )

    if abs(
        face_hw_ratio
    ) < 1e-8:

        raise ValueError(
            "face_hw_ratio is zero"
        )

    # Team face_fortune.py:
    #
    # face_hw_ratio
    # = face_height / face_width
    #
    # Existing animal geometry baseline:
    #
    # face_width_height_ratio
    # = face_width / face_height

    geometry = {

        "face_width_height_ratio":
            1.0 / face_hw_ratio,

        "right_eye_width_ratio":
            float(
                feature_values[
                    "right_eye_width_ratio"
                ]
            ),

        "left_eye_width_ratio":
            float(
                feature_values[
                    "left_eye_width_ratio"
                ]
            ),

        "eye_distance_ratio":
            float(
                feature_values[
                    "eye_distance_ratio"
                ]
            ),

        "mouth_width_ratio":
            float(
                feature_values[
                    "mouth_width_ratio"
                ]
            ),

        "jaw_width_ratio":
            float(
                feature_values[
                    "jaw_width_ratio"
                ]
            ),
    }

    return geometry


# ==================================================
# Geometry scores
# ==================================================

def get_geometry_scores(
    feature_values,
    baseline_mean=None,
    baseline_std=None,
):

    if baseline_mean is None:
        baseline_mean = geometry_mean
    if baseline_std is None:
        baseline_std = geometry_std

    raw_geometry = (
        team_features_to_geometry(
            feature_values
        )
    )

    z_scores = {}

    for (
        key,
        value
    ) in raw_geometry.items():

        if key not in baseline_mean or key not in baseline_std:

            raise KeyError(
                f"Geometry baseline "
                f"does not contain: {key}"
            )

        z_scores[key] = float(
            (
                value
                -
                baseline_mean[key]
            )
            /
            (
                baseline_std[key]
                +
                1e-8
            )
        )

    # Average left/right eye size
    eye_size = (
        z_scores[
            "right_eye_width_ratio"
        ]
        +
        z_scores[
            "left_eye_width_ratio"
        ]
    ) / 2.0

    # This is the vector used by the
    # existing animal geometry model.
    human_vector = np.array(
        [
            z_scores[
                "face_width_height_ratio"
            ],

            eye_size,

            z_scores[
                "eye_distance_ratio"
            ],

            z_scores[
                "mouth_width_ratio"
            ],

            z_scores[
                "jaw_width_ratio"
            ],
        ],
        dtype=np.float32
    )

    human_vector = np.clip(
        human_vector,
        -2.5,
        2.5
    )

    scores = {}
    diagnostics = {}

    for (
        animal,
        target
    ) in (
        animal_geometry_profiles.items()
    ):

        target = np.asarray(
            target,
            dtype=np.float32
        )

        distance = float(
            np.mean(
                (
                    human_vector
                    -
                    target
                )
                ** 2
            )
        )

        similarity = float(
            np.exp(
                -distance
                /
                2.0
            )
        )
        scores[animal] = similarity
        diagnostics[animal] = {
            "distance_mse": distance,
            "similarity": similarity,
        }

    return (
        scores,
        z_scores,
        diagnostics,
    )


# ==================================================
# Animal prediction
# ==================================================

def predict_animal_from_team(
    image_path,
    feature_values,
    multi_frame_image_paths=None,
    multi_frame_face_paths=None,
    multi_frame_face_v41_paths=None,
):

    (
        cnn_scores,
        cnn_diagnostics,
    ) = (
        get_cnn_scores(
            image_path
        )
    )

    (
        geometry_scores,
        z_scores,
        geometry_diagnostics,
    ) = (
        get_geometry_scores(
            feature_values
        )
    )

    cnn_normalized = (
        normalize_scores(
            cnn_scores
        )
    )

    geometry_normalized = (
        normalize_scores(
            geometry_scores
        )
    )

    wflw_ab_scores = None
    wflw_ab_z_scores = None
    wflw_ab_geometry_diagnostics = None
    wflw_ab_geometry_normalized = None
    wflw_ab_runtime_error = wflw_geometry_ab_error
    if wflw_geometry_ab is not None:
        try:
            (
                wflw_ab_scores,
                wflw_ab_z_scores,
                wflw_ab_geometry_diagnostics,
            ) = get_geometry_scores(
                feature_values,
                baseline_mean=wflw_geometry_ab["mean"],
                baseline_std=wflw_geometry_ab["std"],
            )
            wflw_ab_geometry_normalized = normalize_scores(wflw_ab_scores)
        except (KeyError, TypeError, ValueError, FloatingPointError) as error:
            wflw_ab_runtime_error = str(error)

    final_scores = {}

    for animal in cnn_scores:

        final_scores[animal] = (
            0.70
            *
            cnn_normalized[animal]

            +

            0.30
            *
            geometry_normalized[
                animal
            ]
        )

    ranking = sorted(
        final_scores.items(),
        key=lambda item:
            item[1],
        reverse=True
    )[:3]

    values = np.array(
        [
            max(
                score,
                0.0
            )
            for _, score
            in ranking
        ],
        dtype=np.float32
    )

    percentages = (
        values
        /
        (
            values.sum()
            +
            1e-8
        )
        *
        100.0
    )

    if wflw_ab_geometry_normalized is not None:
        wflw_ab_final_scores = {
            animal: (
                0.70 * cnn_normalized[animal]
                + 0.30 * wflw_ab_geometry_normalized[animal]
            )
            for animal in cnn_scores
        }
        wflw_ab_sorted = sorted(
            wflw_ab_final_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        wflw_ab_top3 = wflw_ab_sorted[:3]
        wflw_ab_values = np.array(
            [max(score, 0.0) for _, score in wflw_ab_top3],
            dtype=np.float32,
        )
        wflw_ab_percentages = (
            wflw_ab_values / (wflw_ab_values.sum() + 1e-8) * 100.0
        )
        wflw_ab_rank_lookup = {
            animal: index
            for index, (animal, _) in enumerate(wflw_ab_sorted, start=1)
        }
        wflw_ab_percent_lookup = {
            animal: float(percent)
            for (animal, _), percent in zip(wflw_ab_top3, wflw_ab_percentages)
        }
        wflw_ab_per_animal = {
            animal: {
                "geometry_distance_mse": float(
                    wflw_ab_geometry_diagnostics[animal]["distance_mse"]
                ),
                "geometry_similarity": float(
                    wflw_ab_geometry_diagnostics[animal]["similarity"]
                ),
                "geometry_normalized": float(
                    wflw_ab_geometry_normalized[animal]
                ),
                "final_score": float(final_score),
                "final_rank": int(wflw_ab_rank_lookup[animal]),
                "top3_display_percent": wflw_ab_percent_lookup.get(animal),
            }
            for animal, final_score in wflw_ab_sorted
        }
        wflw_geometry_ab_diagnostics = {
            "status": "available",
            "behavior_changed": False,
            "selection_mode": "diagnostic_only; legacy ranking still selects Pokemon",
            "baseline": {
                key: value
                for key, value in wflw_geometry_ab.items()
                if key not in {"mean", "std"}
            },
            "baseline_mean": wflw_geometry_ab["mean"],
            "baseline_std": wflw_geometry_ab["std"],
            "geometry_z_scores": wflw_ab_z_scores,
            "top1_top2_final_score_margin": (
                float(wflw_ab_sorted[0][1] - wflw_ab_sorted[1][1])
                if len(wflw_ab_sorted) >= 2 else None
            ),
            "top3_top4_final_score_margin": (
                float(wflw_ab_sorted[2][1] - wflw_ab_sorted[3][1])
                if len(wflw_ab_sorted) >= 4 else None
            ),
            "top3": [
                {
                    "rank": index,
                    "animal": animal,
                    "final_score": float(score),
                    "display_percent": float(percent),
                }
                for index, ((animal, score), percent) in enumerate(
                    zip(wflw_ab_top3, wflw_ab_percentages), start=1
                )
            ],
            "per_animal": wflw_ab_per_animal,
        }
    else:
        wflw_geometry_ab_diagnostics = {
            "status": "unavailable",
            "behavior_changed": False,
            "reason": wflw_ab_runtime_error or "WFLW A/B baseline unavailable",
            "top3": [],
            "per_animal": {},
        }

    # Candidate v2 keeps the same WFLW geometry baseline but reduces the
    # geometry contribution from 30% to 25%.  P01-P23 were used for
    # development and P24-P30 for independent validation.  It remains
    # diagnostic-only until a separate follow-up cohort confirms it.
    if wflw_ab_geometry_normalized is not None:
        v2_cnn_weight = 0.75
        v2_geometry_weight = 0.25
        v2_final_scores = {
            animal: (
                v2_cnn_weight * cnn_normalized[animal]
                + v2_geometry_weight * wflw_ab_geometry_normalized[animal]
            )
            for animal in cnn_scores
        }
        v2_sorted = sorted(
            v2_final_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        v2_top3 = v2_sorted[:3]
        v2_values = np.array(
            [max(score, 0.0) for _, score in v2_top3],
            dtype=np.float32,
        )
        v2_percentages = v2_values / (v2_values.sum() + 1e-8) * 100.0
        v2_rank_lookup = {
            animal: index
            for index, (animal, _) in enumerate(v2_sorted, start=1)
        }
        v2_percent_lookup = {
            animal: float(percent)
            for (animal, _), percent in zip(v2_top3, v2_percentages)
        }
        wflw_geometry_v2_diagnostics = {
            "status": "available",
            "behavior_changed": False,
            "selection_mode": "diagnostic_only; legacy ranking still selects Pokemon",
            "candidate_id": "wflw_geometry_75cnn_25geometry_v2",
            "fusion_weights": {
                "cnn": v2_cnn_weight,
                "geometry": v2_geometry_weight,
            },
            "baseline": {
                key: value
                for key, value in wflw_geometry_ab.items()
                if key not in {"mean", "std"}
            },
            "baseline_mean": wflw_geometry_ab["mean"],
            "baseline_std": wflw_geometry_ab["std"],
            "geometry_z_scores": wflw_ab_z_scores,
            "top1_top2_final_score_margin": (
                float(v2_sorted[0][1] - v2_sorted[1][1])
                if len(v2_sorted) >= 2 else None
            ),
            "top3_top4_final_score_margin": (
                float(v2_sorted[2][1] - v2_sorted[3][1])
                if len(v2_sorted) >= 4 else None
            ),
            "top3": [
                {
                    "rank": index,
                    "animal": animal,
                    "final_score": float(score),
                    "display_percent": float(percent),
                }
                for index, ((animal, score), percent) in enumerate(
                    zip(v2_top3, v2_percentages), start=1
                )
            ],
            "per_animal": {
                animal: {
                    "geometry_distance_mse": float(
                        wflw_ab_geometry_diagnostics[animal]["distance_mse"]
                    ),
                    "geometry_similarity": float(
                        wflw_ab_geometry_diagnostics[animal]["similarity"]
                    ),
                    "geometry_normalized": float(
                        wflw_ab_geometry_normalized[animal]
                    ),
                    "final_score": float(final_score),
                    "final_rank": int(v2_rank_lookup[animal]),
                    "top3_display_percent": v2_percent_lookup.get(animal),
                }
                for animal, final_score in v2_sorted
            },
        }
    else:
        wflw_geometry_v2_diagnostics = {
            "status": "unavailable",
            "behavior_changed": False,
            "candidate_id": "wflw_geometry_75cnn_25geometry_v2",
            "reason": wflw_ab_runtime_error or "WFLW A/B baseline unavailable",
            "top3": [],
            "per_animal": {},
        }

    # Candidate v3 averages CNN embeddings from five accepted frames.  The
    # geometry channel is reduced to 15% because the same-person diagnostic
    # showed that the CNN and hand-authored geometry profiles can disagree.
    multi_frame_paths = list(multi_frame_image_paths or [])
    if len(multi_frame_paths) >= 2 and wflw_ab_geometry_normalized is not None:
        try:
            (
                v3_cnn_scores,
                v3_cnn_diagnostics,
                v3_cnn_aggregation,
            ) = get_cnn_scores_multi(multi_frame_paths)
            v3_cnn_normalized = normalize_scores(v3_cnn_scores)
            v3_cnn_weight = 0.85
            v3_geometry_weight = 0.15
            v3_final_scores = {
                animal: (
                    v3_cnn_weight * v3_cnn_normalized[animal]
                    + v3_geometry_weight * wflw_ab_geometry_normalized[animal]
                )
                for animal in v3_cnn_scores
            }
            v3_sorted = sorted(
                v3_final_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            v3_top3 = v3_sorted[:3]
            v3_values = np.array(
                [max(score, 0.0) for _, score in v3_top3],
                dtype=np.float32,
            )
            v3_percentages = v3_values / (v3_values.sum() + 1e-8) * 100.0
            v3_rank_lookup = {
                animal: index
                for index, (animal, _) in enumerate(v3_sorted, start=1)
            }
            v3_percent_lookup = {
                animal: float(percent)
                for (animal, _), percent in zip(v3_top3, v3_percentages)
            }
            wflw_multiframe_v3_diagnostics = {
                "status": "available",
                "behavior_changed": False,
                "selection_mode": "diagnostic_only; legacy ranking still selects Pokemon",
                "candidate_id": "wflw_multiframe_85cnn_15geometry_v3",
                "fusion_weights": {
                    "cnn": v3_cnn_weight,
                    "geometry": v3_geometry_weight,
                },
                "cnn_aggregation": v3_cnn_aggregation,
                "baseline": {
                    key: value
                    for key, value in wflw_geometry_ab.items()
                    if key not in {"mean", "std"}
                },
                "geometry_z_scores": wflw_ab_z_scores,
                "top1_top2_final_score_margin": (
                    float(v3_sorted[0][1] - v3_sorted[1][1])
                    if len(v3_sorted) >= 2 else None
                ),
                "top3_top4_final_score_margin": (
                    float(v3_sorted[2][1] - v3_sorted[3][1])
                    if len(v3_sorted) >= 4 else None
                ),
                "top3": [
                    {
                        "rank": index,
                        "animal": animal,
                        "final_score": float(score),
                        "display_percent": float(percent),
                    }
                    for index, ((animal, score), percent) in enumerate(
                        zip(v3_top3, v3_percentages), start=1
                    )
                ],
                "per_animal": {
                    animal: {
                        **v3_cnn_diagnostics[animal],
                        "cnn_normalized": float(v3_cnn_normalized[animal]),
                        "geometry_distance_mse": float(
                            wflw_ab_geometry_diagnostics[animal]["distance_mse"]
                        ),
                        "geometry_similarity": float(
                            wflw_ab_geometry_diagnostics[animal]["similarity"]
                        ),
                        "geometry_normalized": float(
                            wflw_ab_geometry_normalized[animal]
                        ),
                        "final_score": float(final_score),
                        "final_rank": int(v3_rank_lookup[animal]),
                        "top3_display_percent": v3_percent_lookup.get(animal),
                    }
                    for animal, final_score in v3_sorted
                },
            }
        except (OSError, RuntimeError, TypeError, ValueError, FloatingPointError) as error:
            wflw_multiframe_v3_diagnostics = {
                "status": "unavailable",
                "behavior_changed": False,
                "candidate_id": "wflw_multiframe_85cnn_15geometry_v3",
                "reason": str(error),
                "top3": [],
                "per_animal": {},
            }
    else:
        if len(multi_frame_paths) < 2:
            v3_reason = (
                "At least two cnn_frame_*.jpg images are required; "
                f"found {len(multi_frame_paths)}"
            )
        else:
            v3_reason = wflw_ab_runtime_error or "WFLW geometry baseline unavailable"
        wflw_multiframe_v3_diagnostics = {
            "status": "unavailable",
            "behavior_changed": False,
            "candidate_id": "wflw_multiframe_85cnn_15geometry_v3",
            "reason": v3_reason,
            "top3": [],
            "per_animal": {},
        }

    # Candidate v4 uses the same multi-frame aggregation and weights as v3,
    # but every CNN input is a consistently aligned square face crop instead
    # of the full camera frame.  This removes background and face-position
    # changes from the diagnostic candidate.
    multi_face_paths = list(multi_frame_face_paths or [])
    if len(multi_face_paths) >= 2 and wflw_ab_geometry_normalized is not None:
        try:
            (
                v4_cnn_scores,
                v4_cnn_diagnostics,
                v4_cnn_aggregation,
            ) = get_cnn_scores_multi(multi_face_paths)
            v4_cnn_normalized = normalize_scores(v4_cnn_scores)
            v4_cnn_weight = 0.85
            v4_geometry_weight = 0.15
            v4_final_scores = {
                animal: (
                    v4_cnn_weight * v4_cnn_normalized[animal]
                    + v4_geometry_weight * wflw_ab_geometry_normalized[animal]
                )
                for animal in v4_cnn_scores
            }
            v4_sorted = sorted(
                v4_final_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            v4_top3 = v4_sorted[:3]
            v4_values = np.array(
                [max(score, 0.0) for _, score in v4_top3],
                dtype=np.float32,
            )
            v4_percentages = v4_values / (v4_values.sum() + 1e-8) * 100.0
            v4_rank_lookup = {
                animal: index
                for index, (animal, _) in enumerate(v4_sorted, start=1)
            }
            v4_percent_lookup = {
                animal: float(percent)
                for (animal, _), percent in zip(v4_top3, v4_percentages)
            }
            wflw_facecrop_multiframe_v4_diagnostics = {
                "status": "available",
                "behavior_changed": False,
                "selection_mode": "diagnostic_only; legacy ranking still selects Pokemon",
                "candidate_id": "wflw_facecrop_multiframe_85cnn_15geometry_v4",
                "cnn_input": "five square WFLW face crops; margin 1.35",
                "fusion_weights": {
                    "cnn": v4_cnn_weight,
                    "geometry": v4_geometry_weight,
                },
                "cnn_aggregation": v4_cnn_aggregation,
                "baseline": {
                    key: value
                    for key, value in wflw_geometry_ab.items()
                    if key not in {"mean", "std"}
                },
                "geometry_z_scores": wflw_ab_z_scores,
                "top1_top2_final_score_margin": (
                    float(v4_sorted[0][1] - v4_sorted[1][1])
                    if len(v4_sorted) >= 2 else None
                ),
                "top3_top4_final_score_margin": (
                    float(v4_sorted[2][1] - v4_sorted[3][1])
                    if len(v4_sorted) >= 4 else None
                ),
                "top3": [
                    {
                        "rank": index,
                        "animal": animal,
                        "final_score": float(score),
                        "display_percent": float(percent),
                    }
                    for index, ((animal, score), percent) in enumerate(
                        zip(v4_top3, v4_percentages), start=1
                    )
                ],
                "per_animal": {
                    animal: {
                        **v4_cnn_diagnostics[animal],
                        "cnn_normalized": float(v4_cnn_normalized[animal]),
                        "geometry_distance_mse": float(
                            wflw_ab_geometry_diagnostics[animal]["distance_mse"]
                        ),
                        "geometry_similarity": float(
                            wflw_ab_geometry_diagnostics[animal]["similarity"]
                        ),
                        "geometry_normalized": float(
                            wflw_ab_geometry_normalized[animal]
                        ),
                        "final_score": float(final_score),
                        "final_rank": int(v4_rank_lookup[animal]),
                        "top3_display_percent": v4_percent_lookup.get(animal),
                    }
                    for animal, final_score in v4_sorted
                },
            }
        except (OSError, RuntimeError, TypeError, ValueError, FloatingPointError) as error:
            wflw_facecrop_multiframe_v4_diagnostics = {
                "status": "unavailable",
                "behavior_changed": False,
                "candidate_id": "wflw_facecrop_multiframe_85cnn_15geometry_v4",
                "reason": str(error),
                "top3": [],
                "per_animal": {},
            }
    else:
        if len(multi_face_paths) < 2:
            v4_reason = (
                "At least two cnn_face_*.jpg images are required; "
                f"found {len(multi_face_paths)}"
            )
        else:
            v4_reason = wflw_ab_runtime_error or "WFLW geometry baseline unavailable"
        wflw_facecrop_multiframe_v4_diagnostics = {
            "status": "unavailable",
            "behavior_changed": False,
            "candidate_id": "wflw_facecrop_multiframe_85cnn_15geometry_v4",
            "reason": v4_reason,
            "top3": [],
            "per_animal": {},
        }

    # Candidate v4.1 keeps V4 intact but uses five face crops selected by
    # sharpness within five temporal bins of the 30 accepted frames.
    # It remains diagnostic-only and does not affect Pokemon selection.
    multi_face_v41_paths = list(multi_frame_face_v41_paths or [])
    if len(multi_face_v41_paths) >= 2 and wflw_ab_geometry_normalized is not None:
        try:
            (
                v41_cnn_scores,
                v41_cnn_diagnostics,
                v41_cnn_aggregation,
            ) = get_cnn_scores_multi(multi_face_v41_paths)
            v41_cnn_normalized = normalize_scores(v41_cnn_scores)
            v41_cnn_weight = 0.60
            v41_geometry_weight = 0.40
            v41_final_scores = {
                animal: (
                    v41_cnn_weight * v41_cnn_normalized[animal]
                    + v41_geometry_weight * wflw_ab_geometry_normalized[animal]
                )
                for animal in v41_cnn_scores
            }
            v41_sorted = sorted(
                v41_final_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            v41_top3 = v41_sorted[:3]
            # Independent animal MATCH score.
            # This is NOT a probability and TOP3 does NOT sum to 100.
            # Each animal is displayed independently on a 0~100 scale
            # from the fused V4.1 score.
            v41_percentages = np.array(
                [
                    float(np.clip(score, 0.0, 1.0) * 100.0)
                    for _, score in v41_top3
                ],
                dtype=np.float32,
            )
            v41_rank_lookup = {
                animal: index
                for index, (animal, _) in enumerate(v41_sorted, start=1)
            }
            v41_percent_lookup = {
                animal: float(percent)
                for (animal, _), percent in zip(v41_top3, v41_percentages)
            }
            wflw_facecrop_sharp_v41_diagnostics = {
                "status": "available",
                "behavior_changed": True,
                "selection_mode": "production; balanced V4.1 ranking selects Pokemon",
                "candidate_id": "wflw_facecrop_sharp_60cnn_40geometry_v41_balanced",
                "cnn_input": "five square WFLW face crops; sharpest crop from each temporal bin; margin 1.35",
                "fusion_weights": {
                    "cnn": v41_cnn_weight,
                    "geometry": v41_geometry_weight,
                },
                "cnn_aggregation": v41_cnn_aggregation,
                "baseline": {
                    key: value
                    for key, value in wflw_geometry_ab.items()
                    if key not in {"mean", "std"}
                },
                "geometry_z_scores": wflw_ab_z_scores,
                "top1_top2_final_score_margin": (
                    float(v41_sorted[0][1] - v41_sorted[1][1])
                    if len(v41_sorted) >= 2 else None
                ),
                "top3_top4_final_score_margin": (
                    float(v41_sorted[2][1] - v41_sorted[3][1])
                    if len(v41_sorted) >= 4 else None
                ),
                "top3": [
                    {
                        "rank": index,
                        "animal": animal,
                        "final_score": float(score),
                        "match_score": float(percent),
                        "display_percent": float(percent),
                    }
                    for index, ((animal, score), percent) in enumerate(
                        zip(v41_top3, v41_percentages), start=1
                    )
                ],
                "per_animal": {
                    animal: {
                        **v41_cnn_diagnostics[animal],
                        "cnn_normalized": float(v41_cnn_normalized[animal]),
                        "geometry_distance_mse": float(
                            wflw_ab_geometry_diagnostics[animal]["distance_mse"]
                        ),
                        "geometry_similarity": float(
                            wflw_ab_geometry_diagnostics[animal]["similarity"]
                        ),
                        "geometry_normalized": float(
                            wflw_ab_geometry_normalized[animal]
                        ),
                        "final_score": float(final_score),
                        "match_score": float(
                            np.clip(final_score, 0.0, 1.0) * 100.0
                        ),
                        "final_rank": int(v41_rank_lookup[animal]),
                        "top3_display_percent": v41_percent_lookup.get(animal),
                    }
                    for animal, final_score in v41_sorted
                },
            }
        except (OSError, RuntimeError, TypeError, ValueError, FloatingPointError) as error:
            wflw_facecrop_sharp_v41_diagnostics = {
                "status": "unavailable",
                "behavior_changed": False,
                "candidate_id": "wflw_facecrop_sharp_60cnn_40geometry_v41_balanced",
                "reason": str(error),
                "top3": [],
                "per_animal": {},
            }
    else:
        if len(multi_face_v41_paths) < 2:
            v41_reason = (
                "At least two cnn_face_v41_*.jpg images are required; "
                f"found {len(multi_face_v41_paths)}"
            )
        else:
            v41_reason = wflw_ab_runtime_error or "WFLW geometry baseline unavailable"
        wflw_facecrop_sharp_v41_diagnostics = {
            "status": "unavailable",
            "behavior_changed": False,
            "candidate_id": "wflw_facecrop_sharp_85cnn_15geometry_v41",
            "reason": v41_reason,
            "top3": [],
            "per_animal": {},
        }

    # Candidate v4.2 reuses the exact same sharpness-selected face crops
    # as v4.1, but reduces WFLW geometry from 15% to 5%.
    # It remains diagnostic-only and does not affect Pokemon selection.
    if len(multi_face_v41_paths) >= 2 and wflw_ab_geometry_normalized is not None:
        try:
            (
                v42_cnn_scores,
                v42_cnn_diagnostics,
                v42_cnn_aggregation,
            ) = get_cnn_scores_multi(multi_face_v41_paths)
            v42_cnn_normalized = normalize_scores(v42_cnn_scores)
            v42_cnn_weight = 0.95
            v42_geometry_weight = 0.05
            v42_final_scores = {
                animal: (
                    v42_cnn_weight * v42_cnn_normalized[animal]
                    + v42_geometry_weight * wflw_ab_geometry_normalized[animal]
                )
                for animal in v42_cnn_scores
            }
            v42_sorted = sorted(
                v42_final_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            v42_top3 = v42_sorted[:3]
            v42_values = np.array(
                [max(score, 0.0) for _, score in v42_top3],
                dtype=np.float32,
            )
            v42_percentages = v42_values / (v42_values.sum() + 1e-8) * 100.0
            v42_rank_lookup = {
                animal: index
                for index, (animal, _) in enumerate(v42_sorted, start=1)
            }
            v42_percent_lookup = {
                animal: float(percent)
                for (animal, _), percent in zip(v42_top3, v42_percentages)
            }
            wflw_facecrop_sharp_v42_diagnostics = {
                "status": "available",
                "behavior_changed": False,
                "selection_mode": "diagnostic_only; legacy ranking still selects Pokemon",
                "candidate_id": "wflw_facecrop_sharp_95cnn_05geometry_v42",
                "cnn_input": "same five sharpness-selected square WFLW face crops as V4.1; margin 1.35",
                "fusion_weights": {
                    "cnn": v42_cnn_weight,
                    "geometry": v42_geometry_weight,
                },
                "cnn_aggregation": v42_cnn_aggregation,
                "baseline": {
                    key: value
                    for key, value in wflw_geometry_ab.items()
                    if key not in {"mean", "std"}
                },
                "geometry_z_scores": wflw_ab_z_scores,
                "top1_top2_final_score_margin": (
                    float(v42_sorted[0][1] - v42_sorted[1][1])
                    if len(v42_sorted) >= 2 else None
                ),
                "top3_top4_final_score_margin": (
                    float(v42_sorted[2][1] - v42_sorted[3][1])
                    if len(v42_sorted) >= 4 else None
                ),
                "top3": [
                    {
                        "rank": index,
                        "animal": animal,
                        "final_score": float(score),
                        "display_percent": float(percent),
                    }
                    for index, ((animal, score), percent) in enumerate(
                        zip(v42_top3, v42_percentages), start=1
                    )
                ],
                "per_animal": {
                    animal: {
                        **v42_cnn_diagnostics[animal],
                        "cnn_normalized": float(v42_cnn_normalized[animal]),
                        "geometry_distance_mse": float(
                            wflw_ab_geometry_diagnostics[animal]["distance_mse"]
                        ),
                        "geometry_similarity": float(
                            wflw_ab_geometry_diagnostics[animal]["similarity"]
                        ),
                        "geometry_normalized": float(
                            wflw_ab_geometry_normalized[animal]
                        ),
                        "final_score": float(final_score),
                        "final_rank": int(v42_rank_lookup[animal]),
                        "top3_display_percent": v42_percent_lookup.get(animal),
                    }
                    for animal, final_score in v42_sorted
                },
            }
        except (OSError, RuntimeError, TypeError, ValueError, FloatingPointError) as error:
            wflw_facecrop_sharp_v42_diagnostics = {
                "status": "unavailable",
                "behavior_changed": False,
                "candidate_id": "wflw_facecrop_sharp_95cnn_05geometry_v42",
                "reason": str(error),
                "top3": [],
                "per_animal": {},
            }
    else:
        if len(multi_face_v41_paths) < 2:
            v42_reason = (
                "At least two cnn_face_v41_*.jpg images are required; "
                f"found {len(multi_face_v41_paths)}"
            )
        else:
            v42_reason = wflw_ab_runtime_error or "WFLW geometry baseline unavailable"
        wflw_facecrop_sharp_v42_diagnostics = {
            "status": "unavailable",
            "behavior_changed": False,
            "candidate_id": "wflw_facecrop_sharp_95cnn_05geometry_v42",
            "reason": v42_reason,
            "top3": [],
            "per_animal": {},
        }

    sorted_all = sorted(
        final_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    rank_lookup = {
        animal: index
        for index, (animal, _) in enumerate(sorted_all, start=1)
    }
    top3_percent_lookup = {
        animal: float(percent)
        for (animal, _), percent in zip(ranking, percentages)
    }

    per_animal = {}
    for animal, final_score in sorted_all:
        per_animal[animal] = {
            **cnn_diagnostics[animal],
            "cnn_normalized": float(cnn_normalized[animal]),
            "geometry_distance_mse": float(
                geometry_diagnostics[animal]["distance_mse"]
            ),
            "geometry_similarity": float(
                geometry_diagnostics[animal]["similarity"]
            ),
            "geometry_normalized": float(geometry_normalized[animal]),
            "final_score": float(final_score),
            "final_rank": int(rank_lookup[animal]),
            "top3_display_percent": top3_percent_lookup.get(animal),
        }

    score_diagnostics = {
        "schema_version": 1,
        "behavior_changed": False,
        "animal_count": len(sorted_all),
        "cnn_model_input": {
            "shape": [int(value) for value in input_details[0]["shape"]],
            "dtype": str(input_details[0]["dtype"]),
            "quantization_scale": float(input_details[0]["quantization"][0]),
            "quantization_zero_point": int(input_details[0]["quantization"][1]),
            "preprocessing": "BGR to RGB, resize 224x224, cast float32, no value scaling",
        },
        "fusion_weights": {"cnn": 0.70, "geometry": 0.30},
        "normalization": "independent per-capture min-max for CNN and geometry",
        "display_percent_semantics": (
            "share of the three non-negative final TOP3 scores; "
            "not a calibrated probability"
        ),
        "top1_top2_final_score_margin": (
            float(sorted_all[0][1] - sorted_all[1][1])
            if len(sorted_all) >= 2 else None
        ),
        "top3_top4_final_score_margin": (
            float(sorted_all[2][1] - sorted_all[3][1])
            if len(sorted_all) >= 4 else None
        ),
        "per_animal": per_animal,
        "wflw_geometry_ab": wflw_geometry_ab_diagnostics,
        "wflw_geometry_v2": wflw_geometry_v2_diagnostics,
        "wflw_multiframe_v3": wflw_multiframe_v3_diagnostics,
        "wflw_facecrop_multiframe_v4": wflw_facecrop_multiframe_v4_diagnostics,
        "wflw_facecrop_sharp_v41": wflw_facecrop_sharp_v41_diagnostics,
        "wflw_facecrop_sharp_v42": wflw_facecrop_sharp_v42_diagnostics,
    }

    # Production selection:
    # V4.1 = five sharpness-selected WFLW face crops,
    # mean CNN embedding, CNN 60% + WFLW geometry 40%.
    # If V4.1 is unavailable for any reason, preserve a safe legacy fallback.
    if wflw_facecrop_sharp_v41_diagnostics.get("status") == "available":
        ranking = [
            (item["animal"], float(item["final_score"]))
            for item in wflw_facecrop_sharp_v41_diagnostics["top3"]
        ]
        percentages = np.array(
            [
                float(item["display_percent"])
                for item in wflw_facecrop_sharp_v41_diagnostics["top3"]
            ],
            dtype=np.float32,
        )
        z_scores = dict(wflw_ab_z_scores)
        score_diagnostics["production_selection"] = {
            "status": "v4.1_active",
            "method": (
                "five sharpness-selected WFLW face crops; "
                "mean of L2-normalized CNN embeddings; "
                "CNN 60% + WFLW geometry 40%"
            ),
            "pokemon_uses_this_ranking": True,
            "display_score_semantics": (
                "independent 0-100 MATCH score from clipped fused final_score; "
                "not a probability; TOP3 does not sum to 100"
            ),
            "fallback_used": False,
        }
    else:
        score_diagnostics["production_selection"] = {
            "status": "legacy_fallback",
            "method": "legacy single-image CNN 70% + legacy geometry 30%",
            "pokemon_uses_this_ranking": True,
            "fallback_used": True,
            "reason": wflw_facecrop_sharp_v41_diagnostics.get(
                "reason",
                "V4.1 unavailable",
            ),
        }

    return (
        ranking,
        percentages,
        z_scores,
        score_diagnostics,
    )


# ==================================================
# Pokemon prediction
# ==================================================

def predict_pokemon_from_team(
    ranking,
    percentages,
    z_scores
):

    with POKEMON_CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(
            file
        )

    pokemon_weight_map = (
        config[
            "pokemon_weight_map"
        ]
    )

    pokemon_style_profile = (
        config[
            "pokemon_style_profile"
        ]
    )

    # Same function used by the existing
    # test_pokemon_final.py.
    human_style = (
        get_style_scores(
            z_scores
        )
    )

    results = []

    for (
        (animal, _),
        percent
    ) in zip(
        ranking,
        percentages
    ):

        (
            pokemon,
            pokemon_score
        ) = (
            select_pokemon(

                animal,

                float(
                    percent
                ),

                human_style,

                pokemon_weight_map,

                pokemon_style_profile
            )
        )

        results.append(
            {
                "animal":
                    animal,

                # Backward-compatible field name. Its meaning is now an
                # independent MATCH score, not a TOP3 share/probability.
                "animal_percent":
                    float(
                        percent
                    ),

                "animal_match_score":
                    float(
                        percent
                    ),

                "pokemon":
                    pokemon,

                "pokemon_score":
                    float(
                        pokemon_score
                    ),
            }
        )

    return (
        results,
        human_style
    )

def analyze_capture_folder(capture_dir):

    capture_dir = Path(capture_dir)

    image_path = (
        capture_dir
        / "face_measurement.jpg"
    )

    handoff_path = (
        capture_dir
        / "handoff.json"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not handoff_path.exists():
        raise FileNotFoundError(
            f"Handoff not found: {handoff_path}"
        )

    with handoff_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        handoff = json.load(file)

    quality = handoff.get(
        "quality",
        {}
    )

    if not quality.get(
        "stable",
        False
    ):
        raise RuntimeError(
            "Measurement is not stable."
        )

    feature_values = handoff[
        "feature_values"
    ]

    multi_frame_image_paths = sorted(
        capture_dir.glob("cnn_frame_*.jpg")
    )
    multi_frame_face_paths = sorted(
        path for path in capture_dir.glob("cnn_face_*.jpg")
        if not path.name.startswith("cnn_face_v41_")
    )
    multi_frame_face_v41_paths = sorted(
        capture_dir.glob("cnn_face_v41_*.jpg")
    )

    (
        ranking,
        percentages,
        z_scores,
        score_diagnostics,
    ) = predict_animal_from_team(
        image_path,
        feature_values,
        multi_frame_image_paths=multi_frame_image_paths,
        multi_frame_face_paths=multi_frame_face_paths,
        multi_frame_face_v41_paths=multi_frame_face_v41_paths,
    )

    (
        pokemon_results,
        human_style
    ) = predict_pokemon_from_team(
        ranking,
        percentages,
        z_scores
    )

    return {
        "ranking": ranking,
        "percentages": percentages,
        "z_scores": z_scores,
        "pokemon_results": pokemon_results,
        "human_style": human_style,
        "quality": quality,
        "score_diagnostics": score_diagnostics,
        "animal_score_semantics": (
            "independent 0-100 MATCH score; not probability; TOP3 does not sum to 100"
        ),
    }
# ==================================================
# Main
# ==================================================

def main():

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python "
            "run_team_animal_test.py "
            "<capture_folder>"
        )

        print()

        return

    capture_dir = Path(
        sys.argv[1]
    )

    image_path = (
        capture_dir
        /
        "face_measurement.jpg"
    )

    handoff_path = (
        capture_dir
        /
        "handoff.json"
    )

    # ----------------------------------------------
    # File check
    # ----------------------------------------------

    if not capture_dir.exists():

        raise FileNotFoundError(
            f"Capture folder not found: "
            f"{capture_dir}"
        )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    if not handoff_path.exists():

        raise FileNotFoundError(
            f"Handoff not found: "
            f"{handoff_path}"
        )

    # ----------------------------------------------
    # Load team handoff
    # ----------------------------------------------

    with handoff_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        handoff = json.load(
            file
        )

    quality = handoff.get(
        "quality",
        {}
    )

    # ----------------------------------------------
    # Quality check
    # ----------------------------------------------

    if not quality.get(
        "stable",
        False
    ):

        raise RuntimeError(
            "Measurement is not stable."
        )

    feature_values = handoff.get(
        "feature_values",
        {}
    )

    if not feature_values:

        raise RuntimeError(
            "feature_values are missing."
        )

    # ----------------------------------------------
    # Animal prediction
    # ----------------------------------------------

    (
        ranking,
        percentages,
        z_scores,
        score_diagnostics,
    ) = (
        predict_animal_from_team(
            image_path,
            feature_values
        )
    )

    # ----------------------------------------------
    # Pokemon prediction
    # ----------------------------------------------

    (
        pokemon_results,
        human_style
    ) = (
        predict_pokemon_from_team(
            ranking,
            percentages,
            z_scores
        )
    )

    # ==============================================
    # Output
    # ==============================================

    print()
    print(
        "========================================"
    )

    print(
        "TEAM FACE FORTUNE TEST"
    )

    print(
        "========================================"
    )

    print()
    print(
        "Capture folder:"
    )

    print(
        capture_dir
    )

    # ----------------------------------------------
    # Quality
    # ----------------------------------------------

    print()
    print(
        "=== QUALITY ==="
    )

    print(
        "stable       :",
        quality.get(
            "stable"
        )
    )

    print(
        "sample_count :",
        quality.get(
            "sample_count"
        )
    )

    print(
        "frontal      :",
        quality.get(
            "frontal"
        )
    )

    print(
        "single_face  :",
        quality.get(
            "single_face"
        )
    )

    print(
        "gaze_verified:",
        quality.get(
            "gaze_verified"
        )
    )

    # ----------------------------------------------
    # Team feature values
    # ----------------------------------------------

    print()
    print(
        "=== TEAM GEOMETRY INPUT ==="
    )

    keys = [
        "face_hw_ratio",
        "right_eye_width_ratio",
        "left_eye_width_ratio",
        "eye_distance_ratio",
        "mouth_width_ratio",
        "jaw_width_ratio",
    ]

    for key in keys:

        print(
            f"{key:25s}: "
            f"{feature_values[key]:.6f}"
        )

    # ----------------------------------------------
    # Geometry Z-score
    # ----------------------------------------------

    print()
    print(
        "=== GEOMETRY Z-SCORES ==="
    )

    for (
        key,
        value
    ) in z_scores.items():

        print(
            f"{key:25s}: "
            f"{value:.3f}"
        )

    # ----------------------------------------------
    # Animal
    # ----------------------------------------------

    print()
    print(
        "=== FINAL ANIMAL TOP3 (INDEPENDENT MATCH SCORE) ==="
    )

    for (
        index,
        (
            (animal, _),
            percent
        )
    ) in enumerate(
        zip(
            ranking,
            percentages
        ),
        start=1
    ):

        print(
            f"{index}. "
            f"{animal:10s} "
            f"MATCH {float(percent):.1f}"
        )

    print()
    print(
        "=== SCORE DIAGNOSTICS ==="
    )

    print(
        "TOP1-TOP2 final margin:",
        f"{score_diagnostics['top1_top2_final_score_margin']:.6f}"
    )

    print(
        "TOP3-TOP4 final margin:",
        f"{score_diagnostics['top3_top4_final_score_margin']:.6f}"
    )

    # ----------------------------------------------
    # Human style
    # ----------------------------------------------

    print()
    print(
        "=== HUMAN STYLE ==="
    )

    for (
        key,
        value
    ) in human_style.items():

        print(
            f"{key:10s}: "
            f"{float(value):.3f}"
        )

    # ----------------------------------------------
    # Animal + Pokemon
    # ----------------------------------------------

    print()
    print(
        "=== ANIMAL + POKEMON TOP3 ==="
    )

    for (
        index,
        item
    ) in enumerate(
        pokemon_results,
        start=1
    ):

        print(
            f"{index}. "
            f"{item['animal']:10s} "
            f"MATCH {item['animal_percent']:.1f} "
            f"-> "
            f"{item['pokemon']}"
        )

    print()
    print(
        "========================================"
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "========================================"
    )

    print()


if __name__ == "__main__":

    main()
