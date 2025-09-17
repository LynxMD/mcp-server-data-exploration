# MCP Server Data Exploration - Task List

## Tasks (Active / Deferred)

### Critical Missing Business Logic Tests

### Active/Planned


#### 2. Memory-First Access Strategy - Tests
- [x] **Test Complete Memory-First Access Flow**
  - Immediate return from memory without disk access (Completed)
  - Disk fallback when not in memory (Completed)
  - Not-found returns None without raising (Completed)
  - **Priority**: High
  - **Effort**: Completed

#### 3. Session Eviction and Reloading - Tests
- [x] **Test Session Reloading After Eviction**
  - Reload after manual memory removal (Completed)
  - Reload after eviction due to `memory_max_sessions` (Completed)
  - Verified session becomes resident in memory after access (Completed)
  - **Priority**: Medium
  - **Effort**: Completed

### Deferred (Implement behind feature flags; add telemetry first where applicable)
- [ ] Intelligent Tiering System
  - Add access frequency/recency counters
  - Promotion/demotion with hysteresis and size-aware scoring
  - Comprehensive tests for anti-thrashing and ordering

### Features Not Yet Implemented

- [ ] Continuous Monitoring Loop
  - Background periodic health checks
  - Proactive cleanup on thresholds
  - Alerting integration and cooldowns
  - Unit tests with mocked timers/psutil

- [ ] Predictive Loading System
  - Preload based on observed usage patterns
  - Telemetry-driven thresholds; tests for efficiency and correctness

- [ ] **Write Intelligent Tiering Tests**
  - Test that frequently accessed data stays in memory for fast access
  - Test that infrequently accessed data is stored on disk to save memory
  - Test that data automatically moves between tiers based on access patterns
  - Test that system optimizes for both speed and memory efficiency
  - Test access pattern tracking and analysis
  - **Priority**: Medium
  - **Estimated Effort**: 3-4 hours

### Future (Phase 4 from research-pulse)
- [ ] Redis Integration & Advanced Monitoring
  - Optional Redis-backed tier; keep memory/disk as front/back caches
  - Extended monitoring/observability

## Notes
- Sliding TTL on disk, graceful degradation, eviction loops, memory-first access, and session reload are implemented and fully tested with 100% coverage in core managers.
