# Fitted thresholds

Written by `uv run nav-jev fit --config configs/<dataset>.yaml`. Each JSON records
`tau_expand`, `tau_stop`, `tau_llm`, the beam settings, the `prompt_version` of
`questions.py`, the resolved `jev_model_id`, and `fitted_on` (always a dev split). The
companion `.fit.json` holds the full grid so the choice can be audited.

`nav-jev bench` refuses thresholds whose `fitted_on` names the split under evaluation,
whose `prompt_version` does not match the current `questions.py`, or which lack a model ID.
