"""Pre-persistence credential redaction and accounting-field removal."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from re import Match, Pattern

from benchmarks.tracing.models import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class RedactionResult[T]:
    value: T
    matches: int
    rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PatternRule:
    name: str
    pattern: Pattern[str]
    replacement: str | Callable[[Match[str]], str]


@dataclass(slots=True)
class _RedactionAccumulator:
    matches: int
    rules: list[str]

    def add(self, rule: str, matches: int = 1) -> None:
        if matches < 1:
            return
        self.matches += matches
        if rule not in self.rules:
            self.rules.append(rule)


_STRUCTURED_CREDENTIAL_FIELDS = {
    "apikey",
    "accesstoken",
    "authorization",
    "authorizationheader",
    "authtoken",
    "clientsecret",
    "cookie",
    "credentials",
    "githubtoken",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "secretkey",
    "signedcredential",
}
_STRUCTURED_CREDENTIAL_SUFFIXES = (
    "apikey",
    "accesskey",
    "accesstoken",
    "authorization",
    "authorizationheader",
    "authtoken",
    "clientsecret",
    "cookie",
    "credentials",
    "githubtoken",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "secretaccesskey",
    "secretkey",
    "signedcredential",
)
_STRUCTURED_ACCOUNTING_FIELDS = {
    "accumulatedcost",
    "accumulatedtokenusage",
    "cachereadtokens",
    "cachewritetokens",
    "cachedtokens",
    "completiontokens",
    "cost",
    "costusd",
    "currency",
    "estimatedcost",
    "estimatedcostusd",
    "inputtokens",
    "outputtokens",
    "price",
    "prompttokens",
    "reasoningtokens",
    "tokencount",
    "tokens",
    "totalcost",
    "totalcostusd",
    "totaltokens",
    "usage",
    "usagesummary",
    "usagetometrics",
}
_STRUCTURED_ACCOUNTING_SUFFIXES = tuple(_STRUCTURED_ACCOUNTING_FIELDS)
_CREDENTIAL_NAME_PATTERN = (
    r"(?:[A-Za-z_][A-Za-z0-9_-]*?)?"
    r"(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|"
    r"authorization(?:[_-]?header)?|client[_-]?secret|cookie|credentials|"
    r"github[_-]?token|password|private[_-]?key|refresh[_-]?token|"
    r"secret(?:[_-]?access)?[_-]?key|secret|signed[_-]?credential)"
)
_PATTERN_RULES = (
    _PatternRule(
        "credential.private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
            r"(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
            re.DOTALL,
        ),
        "<redacted:private_key>",
    ),
    _PatternRule(
        "credential.authorization",
        re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
        "<redacted:authorization>",
    ),
    _PatternRule(
        "credential.model_api_key",
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])"),
        "<redacted:model_api_key>",
    ),
    _PatternRule(
        "credential.github_token",
        re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
        "<redacted:github_token>",
    ),
    _PatternRule(
        "credential.cloud_access_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        "<redacted:cloud_access_key>",
    ),
    _PatternRule(
        "credential.uri_password",
        re.compile(
            r"(?P<scheme>[a-z][a-z0-9+.-]*://)"
            r"(?P<username>[^:/@\s]+):(?!<redacted:)"
            r"(?P<password>[^/@\s]+)@",
            re.IGNORECASE,
        ),
        lambda match: (
            f"{match.group('scheme')}{match.group('username')}:<redacted:uri_password>@"
        ),
    ),
    _PatternRule(
        "credential.assignment",
        re.compile(
            rf"(?i)(?<![A-Za-z0-9_])(?P<option>--?)?"
            rf"(?P<name>{_CREDENTIAL_NAME_PATTERN})"
            r"\s*(?:=|\s)\s*(?!<redacted:)"
            r"(?P<quote>['\"]?)(?P<value>[^\s'\"]{4,})(?P=quote)"
        ),
        lambda match: (
            f"{match.group('option') or ''}{match.group('name')}=<redacted:assignment>"
        ),
    ),
)


class Redactor:
    """Sanitize structured values and arbitrary byte streams before storage."""

    def sanitize_json(self, value: JsonValue) -> RedactionResult[JsonValue]:
        accumulator = _RedactionAccumulator(matches=0, rules=[])
        sanitized = self._sanitize_json(value, accumulator)
        return RedactionResult(
            value=sanitized,
            matches=accumulator.matches,
            rules=tuple(accumulator.rules),
        )

    def sanitize_object(self, value: JsonObject) -> RedactionResult[JsonObject]:
        result = self.sanitize_json(value)
        if not isinstance(result.value, dict):
            raise TypeError("Sanitized trace object is not a mapping")
        return RedactionResult(
            value=result.value,
            matches=result.matches,
            rules=result.rules,
        )

    def sanitize_bytes(self, value: bytes) -> RedactionResult[bytes]:
        text = value.decode("utf-8", errors="surrogateescape")
        result = self.sanitize_text(text)
        return RedactionResult(
            value=result.value.encode("utf-8", errors="surrogateescape"),
            matches=result.matches,
            rules=result.rules,
        )

    def sanitize_text(self, value: str) -> RedactionResult[str]:
        accumulator = _RedactionAccumulator(matches=0, rules=[])
        sanitized = self._sanitize_text(value, accumulator)
        return RedactionResult(
            value=sanitized,
            matches=accumulator.matches,
            rules=tuple(accumulator.rules),
        )

    def detect_json(self, value: JsonValue) -> tuple[str, ...]:
        return self.sanitize_json(value).rules

    def detect_bytes(self, value: bytes) -> tuple[str, ...]:
        return self.sanitize_bytes(value).rules

    def _sanitize_json(
        self, value: JsonValue, accumulator: _RedactionAccumulator
    ) -> JsonValue:
        if isinstance(value, dict):
            sanitized: JsonObject = {}
            for key, item in value.items():
                normalized = re.sub(r"[^a-z0-9]", "", key.lower())
                if _matches_field(
                    normalized,
                    _STRUCTURED_CREDENTIAL_FIELDS,
                    _STRUCTURED_CREDENTIAL_SUFFIXES,
                ):
                    accumulator.add("field.credential")
                    continue
                if _matches_field(
                    normalized,
                    _STRUCTURED_ACCOUNTING_FIELDS,
                    _STRUCTURED_ACCOUNTING_SUFFIXES,
                ):
                    accumulator.add("field.accounting")
                    continue
                sanitized[key] = self._sanitize_json(item, accumulator)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_json(item, accumulator) for item in value]
        if isinstance(value, str):
            return self._sanitize_text(value, accumulator)
        return value

    def _sanitize_text(self, value: str, accumulator: _RedactionAccumulator) -> str:
        sanitized = value
        for rule in _PATTERN_RULES:
            sanitized, matches = rule.pattern.subn(rule.replacement, sanitized)
            accumulator.add(rule.name, matches)
        return sanitized


def _matches_field(
    normalized: str,
    exact: set[str],
    suffixes: tuple[str, ...],
) -> bool:
    return normalized in exact or any(
        normalized != suffix and normalized.endswith(suffix) for suffix in suffixes
    )
