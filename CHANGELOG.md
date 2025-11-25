# Changelog

All notable changes to NexumDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-25

### Added
- WHERE clause filtering with full expression evaluation
  - Comparison operators: =, >, <, >=, <=, !=
  - Logical operators: AND, OR
  - Support for Integer, Float, Text, Boolean types
- Natural Language Query interface (ASK command)
  - NLTranslator class with llama-cpp-python support
  - Fallback rule-based translator
  - Schema-aware translation
- Reinforcement Learning query optimizer
  - Q-Learning agent with state/action/reward
  - Epsilon-greedy exploration
  - Automatic learning from query performance
- ExpressionEvaluator for WHERE clause evaluation
- NLTranslator PyO3 bridge integration
- Schema introspection for NL context
- 3 new integration tests for WHERE filtering
- CLI ASK command mode

### Changed
- Updated CLI to v0.2.0 with enhanced command modes
- Extended Python AI engine with translator and RL agent
- Improved PyO3 bridge with NLTranslator export
- Enhanced README with v0.2.0 features

### Fixed
- Parser now correctly handles WHERE clause expressions
- Fixed unused variable warnings in executor

### Dependencies
- Added: llama-cpp-python>=0.2.0
- Added: diskcache (via llama-cpp-python)

### Tests
- 15 unit tests passing (was 11)
- 3 integration tests passing (new)
- Total: 18/18 tests passing

## [0.1.0] - 2025-11-25

### Added
- Initial release with core SQL database functionality
- Storage engine using sled
- SQL parser for CREATE TABLE, INSERT, SELECT
- Query executor with end-to-end workflow
- Catalog for table metadata management
- Semantic caching using sentence-transformers
- PyO3 Rust-Python integration
- CLI REPL interface
- 11 comprehensive unit tests

### Features
- Persistent KV storage
- SQL query execution
- AI-powered semantic caching
- 60x query speedup on cache hits
- Local-only execution (no cloud dependencies)

[0.2.0]: https://github.com/aviralgarg05/NexumDB/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aviralgarg05/NexumDB/releases/tag/v0.1.0
