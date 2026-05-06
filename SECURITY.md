# Security Policy

## Supported Versions

We currently provide security updates for the latest minor release series.

## Reporting a Vulnerability

Please report vulnerabilities privately via repository security advisories.
Do not open public issues for unpatched vulnerabilities.

## Audit Timestamp Integrity

Audit events include a `timestamp_integrity` boolean in payloads.

- `true`: event timestamp is timezone-aware UTC and non-decreasing in-process.
- `false`: integrity validation failed for that event.

On validation failure, the system emits a dedicated
`audit.timestamp_integrity_warning` event with violation details.
