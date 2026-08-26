from stability_ttt.metrics import compute


def record(seed, condition, success, errors, harmful_at=None, deltas=None):
    return {
        "suite": "libero_spatial",
        "task_id": 0,
        "seed": seed,
        "condition": condition,
        "shift_id": "camera_roll_5",
        "success": success,
        "steps": [
            {
                "t": i,
                "error": error,
                "harmful_update": i == harmful_at,
                "action_delta_l2": (deltas or [0] * len(errors))[i],
            }
            for i, error in enumerate(errors)
        ],
    }


def test_paired_rates_and_propagation():
    records = [
        record(0, "frozen", True, [False, False, False]),
        record(0, "online_persistent", False, [False, True, True], harmful_at=0, deltas=[0.0, 0.2, 0.2]),
        record(1, "frozen", False, [True, True, True]),
        record(1, "online_persistent", True, [True, False, False], deltas=[0.2, 0.2, 0.0]),
    ]
    result = compute(records, [1, 2], 0.01)["conditions"]["online_persistent"]
    assert result["paired_adaptation"]["nar"] == 1.0
    assert result["paired_adaptation"]["par"] == 1.0
    assert result["propagation"]["horizons"]["1"]["conditional_error_probability"] == 1.0
    assert result["update_necessity"]["changed_decision_n"] == 4
