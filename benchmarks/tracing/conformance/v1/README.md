# Benchmark Trace v1 conformance fixtures

The fixtures are the language-neutral acceptance corpus for every recorder and
adapter.

`valid/` is a complete, finalized one-instance trace. Its event stream includes a
shell command with exact arguments, monotonic duration, full stdout artifact, and
linked framework-native evidence. It uses synthetic identifiers and contains no
provider calls or credentials. The compact fixture intentionally uses the
original one-record-per-artifact representation; conforming readers must also
accept the lossless native chunk representation defined by the v1 vocabulary.

`invalid/index.json` maps each intentionally invalid document to the schema that
must reject it. Implementations must reject every case, although diagnostic text
does not need to match across validators.

A conforming implementation must additionally verify semantic invariants that
JSON Schema cannot express:

- schema bundle digests match the installed schemas;
- event and native-record sequences are contiguous and start at 1;
- capability categories are unique and exhaustive;
- artifact paths, byte lengths, and SHA-256 digests match retained bytes;
- native chunk members match their indexed record identifiers and retained
  content digests;
- all attempt documents agree on trace, run, framework, instance, and attempt
  identity;
- referenced event identifiers and attempt paths exist;
- normalized structured metadata has no usage/cost accounting fields or
  authentication material.

The fixture timestamps and monotonic values are illustrative and must not be
treated as performance baselines.
