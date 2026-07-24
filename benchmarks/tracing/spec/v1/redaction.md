# Benchmark Trace v1 retention and redaction policy

The trace is intended to retain complete observable agent activity, but complete
does not mean indiscriminate. Credentials and authentication material must never
become trace artifacts. Token accounting and cost estimates are outside this
trace contract.

## Required processing order

Adapters apply these stages before any journal, normalized event, native record,
artifact, debug log, or digest is written:

1. Capture only the fields needed by the trace contract. Never copy the complete
   process environment.
2. Remove structured authentication fields, provider usage accounting, and cost
   accounting.
3. Replace credential-like values in retained strings and byte streams.
4. Serialize the sanitized value.
5. Hash and atomically persist the sanitized bytes.

An adapter must apply the same policy to framework-native evidence. A file is not
exempt because it is called "raw"; native means structurally native, not
unsanitized.

## Structured fields

Field matching is case-insensitive and separator-insensitive. Authentication
fields include API keys, access tokens, authorization headers, cookies, client
secrets, private keys, passwords, and signed credentials. Environment capture is
allowlist-only and must exclude values for all authentication fields.

Provider usage and cost fields are omitted from normalized provider metadata.
Examples include prompt/completion/token counts, cached-token counters, prices,
currency, and estimated or reported cost. This rule applies to dedicated
accounting fields, not arbitrary research content: a shell result mentioning a
file named `cost.py`, a benchmark problem discussing tokens, or source code with
similarly named identifiers remains part of the lossless tool artifact.

## Credential values

Value scanning must cover at least:

- authorization schemes followed by credentials;
- common hosted-model, source-control, and cloud access-key formats;
- PEM private-key blocks;
- URI user-info containing a password;
- shell assignments or command arguments whose names identify secret material.

Replacements use a stable category marker such as
`<redacted:authorization>`. They must not contain a hash or prefix of the secret.
The artifact reference records only the rule identifiers and number of matches.

User-supplied benchmark text can resemble a credential accidentally. Safety takes
priority over byte-for-byte retention for a matching span. The redaction is made
visible through artifact metadata and health counters.

## Storage behavior

- Trace directories use mode `0700`; files use `0600`.
- Writes use a temporary file in the destination directory, flush and fsync it,
  set its mode, then atomically rename it.
- Artifacts retain all non-secret bytes. Size alone is not a reason to truncate.
- A sanitizer failure blocks persistence of the affected content and degrades
  trace health. It does not fall back to writing the original.
- No implementation may emit secret-bearing diagnostic text while reporting a
  redaction failure.

## Known boundary

The policy protects newly written trace data. It does not retroactively sanitize
pre-existing framework or benchmark logs outside the trace root. Integrations
should avoid duplicate unsafe logs when feasible, but migration of historical
artifacts is separate work.
