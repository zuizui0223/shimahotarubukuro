# Chapter 3: same-tagged-plant observation-to-seed coverage

Priority: High. Tracking issue: #56, within the field programme in #50.

## Purpose and boundary

This additional audit prevents a population-level pool of unrelated plants from
being described as a complete within-plant measurement chain. It belongs to the
prospective Chapter 3 field extension. Neither the closed Chapter 2 manuscript
nor the current Chapter 3 phenotype estimates are changed.

The existing `izu-core` protocol and six native CSV schemas are the starting
point. Their source snapshot is commit
`420734cbf8db7e42c819e2ba7723ec84eccbb9d1`:

- `docs/EFFECTIVE_POLLINATOR_DEPENDENCY_FIELD_PROTOCOL.md`
- `docs/EFFECTIVE_DEPENDENCY_FIELD_FREEZE.md`
- `docs/field_legitimate_contact_protocol.md`
- `channel_id/effective_pollinator_dependency.py`
- the plant, observation-effort, visitor-contact, SVD, treatment and mature-fruit templates.

The native `dependency_panel_structurally_complete` field is population-level.
Its meaning remains unchanged. The new audit is **additional structural QC**,
not a replacement for native schema validation, raw-data provenance/admission,
statistical design review or causal identification.

## Same plant is not the same flower

Under the existing SVD protocol, the stigma is collected after one confirmed
visit. Use separate flowers on the same maternal plant for SVD, no-visit controls
and reproductive treatments. Never reuse a stigma-collected flower as an
untouched fruit-follow-up flower. A distinct single-visit-to-seed arm would need
its own prespecified, non-destructive follow-up protocol; the current core
panel is not such an arm.

The three core reproductive treatments remain `open_pollinated`,
`bagged_autonomous`, and `supplemental_outcross`. The latter must record a donor
other than the mother. Keep assignment, sexual phase, flower order and losses
in the field allocation record; the added coverage script does not verify
randomization or treatment interference.

## Two explicit schema extensions

Before collection, extend the existing templates with:

| Native channel | Added column | Meaning |
| --- | --- | --- |
| effort | `sampling_block_id` | Prespecified comparable observation/assay block. |
| effort | `effort_purpose` | `routine_observation`, `svd_assay`, or `control_assay`. |
| SVD | `sampling_block_id` | Same declared block as the linked effort/visit. |
| treatments | `sampling_block_id` | Paired reproductive-assay block for the same plant. |

These columns must be included in the versioned raw-bundle freeze, not assigned
post hoc to make unrelated observations appear matched. Missing block IDs or
missing routine-observation declarations are reported as coverage gaps. Block
labels alone do not establish biological or temporal comparability.

Keep all native columns. Visits retain exact `effort_id`, `plant_id`, `flower_id`
and `visitor_group`; fruit rows use native `maternal_id` and `site_id`, linked by
unique `fruit_id` from the treatment manifest. One submitted registry has one
context per globally unique plant ID. For longitudinal repeated field events,
retain separate versioned bundles and an explicit cross-event linkage rather
than duplicating a registry ID with conflicting context in one bundle.

Routine zero-visit windows remain exposure records. Experimental waiting for
one visit must not be silently pooled into natural visit-rate estimation. This
script counts coverage only; it does not estimate either visit rate or FDQ.

## Execution

First run the source-locked native freeze, schema, linkage and admission checks.
Then run this extra check against the same CSV bytes:

```bash
python scripts/audit_tagged_plant_chain.py \
  --plants field_dependency_plant_registry.csv \
  --effort field_observation_effort.csv \
  --visits field_visitor_contact_manifest.csv \
  --svd field_single_visit_pollen_deposition.csv \
  --treatments field_pollination_treatments.csv \
  --fruits field_mature_fruit.csv \
  --output tagged_plant_chain_coverage.json
```

The output fingerprints all six supplied files. This fingerprint is an input
identity record, not proof that a pre-analysis freeze or permissions review
was completed. The script uses only the Python standard library.

## Read the output without promoting claims

`same_plant_terminal_panel_present` requires, in one declared plant/block,
routine usable effort, an exact first-visit/SVD link with supported identity and
scored contact, a same-plant/block no-visit control, and at least one terminal
outcome for each core treatment. Recorded `not_seen` contact and zero SVD are
valid measurements; positive results are not required.

`same_plant_seed_panel_resolved` additionally requires seed resolution for every
terminal flower in the three core treatments. An observed abortion is a
terminal zero; a mature fruit with a missing seed count is unresolved, not zero.
The script distinguishes panel presence from unresolved fruit links and seed
counts and retains `pending`, `lost`, and `damaged` counts.

`population_pooled_only_panels` identifies blocks with all channels in the
population pool but no individual plant carrying the complete terminal panel.
Repeated flowers and blocks do not increase `registered_plants`. Zero-visit
plants and plants with no records remain in the output. **Do not use the
complete-panel subset as an automatic analysis filter**: selective exclusion of
unvisited or incompletely observed plants would change the scientific target.
Predeclared incomplete-block analyses are possible, but their coverage must not
be relabelled as complete within-plant chains.

Invalid reused flowers, cross-plant joins and reused fruit/visit IDs raise errors.
Missing controls, sampling blocks and seed records remain explicit gaps.

Even a complete structural panel leaves these separate:

- empirical provenance and native QC;
- quantitative community-functional-trait coverage and matching;
- adequate independent plant/site/time replication and precision;
- causal mediation, historical transition and adaptive evolution.

`historical_transition_identification` is always `not_assessed`. Documented
partner change, comparable pre/post population states, appropriate temporal or
replicated contrasts, and population-history alternatives need their own
analysis. No contemporary completeness flag validates Chapter 2 or proves
historical Bombus loss. Mature seed count is not seed viability or lifetime
fitness.

## Software validation

The fixtures in `tests/test_tagged_plant_chain.py` are synthetic software tests,
not field data. Run:

```bash
python -m unittest discover -s tests -p 'test_tagged_plant_chain.py' -v
```

They cover disjoint plant sets, different blocks, destructive flower reuse,
wrong mother/fruit joins, missing controls and seed counts, all-zero outcomes,
retained losses and zero-visit plants, unknown identity, and CLI input hashes.

Current empirical state remains **field data not supplied**. Implementing this
check does not close the prospective measurement chain in #56.
