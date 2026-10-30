# Prediction label specification (version 1)

## Unit and timing
A sample is one `(episode_id, step, zone)` decision. Features are collected before the next simulator action. The target window includes future steps `t+1` through `t+w`, inclusive. The current step never labels its own pre-action sample.

## Primary target
`attack_risk_w = 1` if a red attack attempt, a newly appearing red session (compromise), or a damage action occurs in the zone within the next `w` steps. The four reporting categories remain distinct: scanning, attack attempts, compromises, and damage/service disruption. Scanning is not a primary positive event.

## Secondary targets
- `compromise_risk_w`: a new red session appears in the zone within the window.
- `scan_risk_w`: a scan/discovery action targets a host in the zone within the window.
- `damage_risk_w`: a damage/service-disruption action targets a host in the zone within the window.

## Event merging
For event-count and lead-time reporting, events in the same zone and category separated by no more than five steps are merged. The five-step default is fixed before model evaluation. Sensitivity analysis should repeat event-level reporting with gaps 0, 3, 5, and 10. Do not use merged event counts as a substitute for step-level prediction labels.

## Clean and compromised strata
The true-state foothold flag is evaluation-only and must never be a model feature. A clean-zone sample has no red foothold in that zone at decision time; a compromised-zone sample is evaluated as escalation. The current collector records a simplified episode-derived stratum and requires verification against the simulator's true state before reporting results.

## Zones
- `blue_agent_0`: Restricted A (`restricted_zone_a`)
- `blue_agent_1`: Operational A (`operational_zone_a`)
- `blue_agent_2`: Restricted B (`restricted_zone_b`)
- `blue_agent_3`: Operational B (`operational_zone_b`)
- `blue_agent_4`: HQ (`public_access_zone`, `admin_network`, `office_network`)

Contractor-network hosts are not silently mapped to HQ; they remain unmapped until a project owner confirms how those hosts should be handled. Events without an identifiable target host or known zone are not assigned to a zone label.

## Runtime information boundary
A deployed blue agent receives its own zone's risk score only. Other-zone risk may be communicated only through the existing 8-bit messages with the project's trust weighting. The centralized critic may receive all zones' risks during training, but no critic-only data may enter deployed actor inputs. This package does not yet modify MAPPO or its observations.

## Episode length
100 steps is the initial setting, not a claim that it is sufficient. Verify per-window event counts, phase transitions, and number of positive episodes before choosing the final length.
