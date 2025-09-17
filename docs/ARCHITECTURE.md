# Architecture Overview

## Project Overview

This project implements a hybrid data management system for MCP (Model Context Protocol) servers, providing intelligent two-tier storage with memory and filesystem layers.

## Core Components

1. **HybridDataManager**: Orchestrates memory and filesystem tiers
2. **TTLInMemoryDataManager**: In-memory storage with sliding TTL and session caps
3. **DiskCacheDataManager**: Filesystem storage with automatic cleanup and TTL management
4. **StorageTypes**: Data structures for storage statistics and tier management
5. **SessionMetadata**: Session tracking and metadata management

## Design Principles

- **Memory-First Access**: Always attempt memory first for optimal performance
- **Lazy Loading**: Load from disk to memory only when needed and space is available
- **Session-Centric**: All operations organized by session
- **Intelligent Eviction**: Oldest sessions evicted first under pressure
- **Graceful Degradation**: Continue operating if one tier fails
- **Thread Safety**: Concurrent access with locking

## Links
- Tasks: see `TASK.md`
- Business logic: see `docs/HYBRID_DATA_MANAGEMENT_BUSINESS_LOGIC.md`
