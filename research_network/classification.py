from __future__ import annotations

from .config import ClassificationConfig


def classify(metrics: dict, config: ClassificationConfig) -> dict:
    flags = []
    if metrics["spike_rate_network"] >= config.high_activity_rate: flags.append("HIGH_ACTIVITY")
    if metrics["synchrony"] >= config.high_synchrony and metrics["total_spikes"]: flags.append("HIGH_SYNCHRONY")
    final_zero = all(abs(x["final"]) <= config.zero_epsilon for x in metrics["potential_per_neuron"])
    if metrics.get("periodic_activity_detected"):
        period = metrics["full_state_period"] or metrics["spike_period"]
        regime = "TONIC" if period == 1 and metrics["activity_survived_until_end"] else "PERIODIC"
    elif metrics["activity_survived_until_end"]:
        regime = "ACTIVE_APERIODIC"
    elif metrics.get("activity_lifetime", 0) > 0:
        regime = "TRANSIENT"
    elif final_zero:
        regime = "DEAD"
    else:
        regime = "QUIESCENT_WITH_STATE"
    return {"primary_regime": regime, "diagnostic_flags": flags}
