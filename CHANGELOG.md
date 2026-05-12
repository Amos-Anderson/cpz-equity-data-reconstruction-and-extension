# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-12

### Added
- Initial public release of data-construction pipeline
- Stage 0 data reconstruction for 46 fundamental characteristics
- Point-in-time merged accounting data with 380-day tolerance
- Three-arm validation framework (CPZ, Fama-French, internal audits)
- 7 characteristic families: value, profitability, investment, momentum, risk, liquidity, other
- Full test coverage with synthetic fixtures
- WRDS integration for live data pulls and caching
- Comprehensive documentation and equation references

### Features
- Deterministic, idempotent pipeline
- Rank-normalized characteristics in (-0.5, 0.5) range
- Coverage filtering (all 46 characteristics required)
- Structured logging and diagnostic outputs
- CI/CD with GitHub Actions
