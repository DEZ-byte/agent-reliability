# Configuration format

`gates.yaml` uses JSON syntax, which is a valid YAML 1.2 subset. The M0
reliability kernel therefore loads it with Python's strict standard-library
JSON parser and does not need a second serialization dependency.

The loader rejects duplicate keys, non-finite numbers, unsupported schema
versions, unknown predicates, and tool-policy mismatches. A future YAML-only
feature requires a dated decision and a pinned parser dependency.
