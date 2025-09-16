# Hybrid Data Management System - Business Logic Documentation

## Overview

The Hybrid Data Management System is a sophisticated two-tier storage architecture that combines fast memory access with persistent disk storage. It automatically manages data across both tiers based on usage patterns, memory pressure, and time-based expiration rules.

## Core Architecture

### Two-Tier Storage System

**Memory Tier (Fast Access)**
- **Purpose**: Ultra-fast data access for active sessions
- **Capacity**: Limited by available system memory
- **Retention**: 5 hours (configurable)
- **Access Speed**: Microseconds
- **Use Case**: Active data processing and real-time operations

**Filesystem Tier (Persistent Storage)**
- **Purpose**: Long-term data persistence and backup
- **Capacity**: Limited by available disk space
- **Retention**: 7 days (configurable)
- **Access Speed**: Milliseconds
- **Use Case**: Data recovery, session restoration, and long-term storage

## Business Rules and Logic

### 1. Data Writing Rules

**Rule: Always Write to Both Tiers**
- When new data is stored, it is **always** written to both memory and filesystem simultaneously
- This ensures data is immediately available in memory for fast access
- This also ensures data is safely persisted to disk for long-term storage
- No data is ever lost due to memory-only storage

**Rule: Write Order**
1. Check if there's enough memory space for the new data
2. If memory is full (90% threshold), evict oldest sessions from memory first
3. Write data to memory tier
4. Write data to filesystem tier
5. Update size tracking for both tiers

### 2. Data Reading Rules

**Rule: Memory-First Access Strategy**
- Always attempt to read from memory first (fastest access)
- If data exists in memory, return it immediately
- If data is not in memory, check if it exists on disk
- If data exists on disk, load it into memory (lazy loading) and return it
- If data doesn't exist in either tier, return empty/not found

**Rule: Lazy Loading Behavior**
- When data is read from disk, it is automatically loaded back into memory
- This ensures subsequent reads of the same data are fast
- Lazy loading only occurs when there's sufficient memory space
- If memory is full, data is read directly from disk without loading to memory

### 3. Memory Management Rules

**Rule: Memory Pressure Monitoring**
- System continuously monitors memory usage percentage
- When memory usage reaches 90% (configurable threshold), pressure relief is triggered
- Memory pressure is checked before adding new data
- Memory pressure is checked before loading data from disk to memory

**Rule: Intelligent Eviction Strategy**
- When memory pressure is detected, oldest sessions are evicted first
- Sessions are ranked by last access time (oldest first)
- Entire sessions are evicted, not individual data items
- Eviction continues until memory usage drops below safe threshold
- Evicted data remains available on disk

**Rule: Size-Aware Memory Management**
- System tracks the size of each data item and each session
- Before loading data to memory, system checks if there's enough space
- If insufficient space, system attempts to free up space by evicting old sessions
- If still insufficient space, data is accessed directly from disk

### 4. Time-Based Expiration Rules

**Rule: Memory Tier Expiration (5 Hours)**
- Data in memory automatically expires after 5 hours of inactivity
- Expiration is based on last access time, not creation time
- Expired data is automatically removed from memory
- Expired data remains available on disk until disk expiration

**Rule: Filesystem Tier Expiration (7 Days)**
- Data on disk automatically expires after 7 days of inactivity
- Expiration is based on last access time, not creation time
- Expired data is permanently deleted from disk
- Background cleanup process runs periodically to remove expired files

**Rule: Sliding Window Expiration**
- Expiration time is reset every time data is accessed
- Active data never expires as long as it's being used
- Only truly inactive data is removed from the system

### 5. Session-Based Management Rules

**Rule: Session-Centric Operations**
- All data operations are organized by session
- Sessions contain multiple data items (DataFrames, objects, etc.)
- Session operations (load, evict, expire) affect the entire session
- Individual data items within a session cannot be partially evicted

**Rule: Session Loading Strategy**
- When a session is accessed, the entire session is loaded to memory
- This ensures all data in a session is available for fast access
- Session loading only occurs if there's sufficient memory space
- If insufficient space, individual data items are accessed directly from disk

**Rule: Session Eviction Strategy**
- When memory pressure occurs, entire sessions are evicted
- Partial session eviction is not allowed
- Evicted sessions remain available on disk
- Sessions can be reloaded to memory when accessed again

### 6. Error Handling and Fallback Rules

**Rule: Graceful Degradation**
- If memory operations fail, system falls back to disk-only operation
- If disk operations fail, system continues with memory-only operation
- If both tiers fail, system returns appropriate error messages
- System never crashes due to storage failures

**Rule: Data Consistency**
- Data is always consistent between memory and disk tiers
- If memory data becomes corrupted, system falls back to disk data
- If disk data becomes corrupted, system continues with memory data
- System automatically recovers from temporary failures

**Rule: Concurrent Access Safety**
- Multiple threads can safely access the system simultaneously
- Thread-safe locking prevents data corruption
- Concurrent session loading is handled gracefully
- Race conditions are prevented through proper synchronization

### 7. Performance Optimization Rules

**Rule: Intelligent Tiering**
- Frequently accessed data stays in memory for fast access
- Infrequently accessed data is stored on disk to save memory
- Data automatically moves between tiers based on access patterns
- System optimizes for both speed and memory efficiency

**Rule: Predictive Loading**
- System can preload sessions to memory based on usage patterns
- Bulk session loading is more efficient than individual item loading
- System can force-load critical sessions to memory
- Loading strategies adapt to available memory and disk space

**Rule: Resource Monitoring**
- System continuously monitors memory and disk usage
- Proactive cleanup prevents resource exhaustion
- System provides detailed statistics on storage usage
- Performance metrics help optimize system behavior

## Usage Scenarios

### Scenario 1: Normal Operation
1. User stores data → Written to both memory and disk
2. User accesses data → Retrieved from memory (fast)
3. Data remains in memory for subsequent fast access
4. After 5 hours of inactivity → Data expires from memory
5. Data remains on disk for 7 days total

### Scenario 2: Memory Pressure
1. Memory usage reaches 90% → Pressure relief triggered
2. Oldest sessions evicted from memory → Memory freed
3. New data written to both tiers → Operation continues
4. Evicted data remains available on disk
5. When evicted data is accessed → Lazy loaded back to memory

### Scenario 3: Disk-Only Access
1. Memory is full and cannot be freed → New data written to disk only
2. Data access attempts → Direct disk access (slower but functional)
3. When memory becomes available → Data can be loaded to memory
4. System continues to function with reduced performance

### Scenario 4: Session Restoration
1. Session expires from memory → Data remains on disk
2. User accesses session → System loads entire session from disk to memory
3. All session data becomes available for fast access
4. Session remains in memory until next expiration or eviction

### Scenario 5: System Recovery
1. System restarts → Memory is empty, disk data persists
2. User accesses data → System loads from disk to memory
3. Normal operation resumes with restored data
4. No data loss occurs during system restarts

## Configuration Parameters

### Memory Tier Configuration
- **TTL**: 5 hours (configurable)
- **Max Sessions**: 100 (configurable)
- **Max Items per Session**: 50 (configurable)
- **Memory Threshold**: 90% (configurable)

### Filesystem Tier Configuration
- **TTL**: 7 days (configurable)
- **Cache Directory**: /tmp/mcp_cache (configurable)
- **Serialization Format**: Parquet for DataFrames, Pickle for other data
- **Disk Usage Threshold**: 90% (configurable)

### System Configuration
- **Cleanup Interval**: 5 minutes (configurable)
- **Background Threads**: Enabled for automatic cleanup
- **Thread Safety**: Enabled for concurrent access
- **Error Recovery**: Enabled for graceful failure handling

## Benefits of This Architecture

### Performance Benefits
- **Fast Access**: Active data available in memory for microsecond access
- **Persistent Storage**: Data safely stored on disk for long-term access
- **Intelligent Caching**: Frequently used data stays in fast memory
- **Automatic Optimization**: System self-optimizes based on usage patterns

### Reliability Benefits
- **Data Safety**: Data always written to both tiers, no data loss
- **Fault Tolerance**: System continues operating even if one tier fails
- **Automatic Recovery**: System recovers from failures without intervention
- **Consistent State**: Data remains consistent across both tiers

### Scalability Benefits
- **Memory Efficient**: Automatic memory management prevents exhaustion
- **Disk Efficient**: Automatic cleanup prevents disk space exhaustion
- **Session-Based**: Efficient handling of large datasets
- **Configurable**: System adapts to different resource constraints

### Operational Benefits
- **Zero Maintenance**: System automatically manages itself
- **Transparent Operation**: Users don't need to manage storage tiers
- **Predictable Behavior**: Clear rules and consistent operation
- **Production Ready**: Robust error handling and edge case management

## Summary

The Hybrid Data Management System provides a sophisticated, self-managing storage solution that combines the speed of memory with the persistence of disk storage. It automatically handles data tiering, memory management, expiration, and error recovery, ensuring optimal performance and reliability while requiring zero manual intervention.

The system is designed to be production-ready, handling all edge cases gracefully while providing consistent, predictable behavior that scales with system resources and usage patterns.
