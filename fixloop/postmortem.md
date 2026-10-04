# Fix Loop Post-Mortem

- `before` runs successfully reproduced the hypothesis: both reported 2.4 m ceiling with method "conformal".
- `floor_only` correctly fell back to method "prior" with a 0.80 m width (>= 0.79 m predicted).
- `with_ceiling` MISSED the prediction. The pipeline reported a ceiling candidate 0.47 m outside [2.0, 4.5] (meaning the point cloud max height was outside typical residential bounds). It thus correctly fell back to "prior", rather than "measured" with a tight 0.06 m width. This outcome matches the stated risk.
- Accuracy against tape is NOT MEASURED (as we lack a tape reference). The goal of reporting an honest prior with a wide band was achieved.
