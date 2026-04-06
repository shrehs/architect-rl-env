# 🎯 System Design Concept Mapping

## Overview
This document maps the architectural constraints in the simulator to **real system design interview concepts**. The system now covers:

- ✅ **CAP Theorem** (Consistency vs Availability Partition tolerance)
- ✅ **Load Balancing** (Traffic patterns and distribution)
- ✅ **CDN & Replication** (Geographic distribution)
- ✅ **Availability** (Fault tolerance and uptime)
- ✅ **Message Queues** (Async processing and decoupling)
- ✅ **Rate Limiting** (API protection and quota management)
- ✅ **Scalability** (Data volume and growth)

---

## Constraint to Concept Mapping

### 1. **SCALABILITY** ← `data_size`
**System Design Interview Question:**
"How does your system handle 1TB → 100TB of data?"

**Key Architectural Patterns:**
- Database sharding (horizontal partitioning)
- Replication across shards
- Consistent hashing for distributed systems
- Read replicas for scaling reads

**Trade-offs:**
- Sharding complexity
- Cross-shard joins become expensive
- Transaction consistency across shards

**Examples:**
- Easy: "Moderate (100GB-1TB)" → Single database, basic indexing
- Medium: "Large (10TB+)" → Need sharding strategy
- Hard: "Very large (100TB+ models)" → Need distributed storage + computation

---

### 2. **LATENCY ↔ THROUGHPUT TRADEOFF** ← `latency`
**System Design Interview Question:**
"How do you achieve <50ms latency while handling 1M QPS?"

**Key Architectural Patterns:**
- In-memory caching (Redis, Memcached)
- CDN for content delivery
- Connection pooling
- Async processing (queue requests)

**Trade-offs:**
- Caching increases complexity (cache invalidation)
- CDN adds cost
- Replication lag may violate freshness

**Examples:**
- Easy: "100ms acceptable" → Can use database queries + caching
- Medium: "500ms acceptable" → Need event streaming
- Hard: "<50ms p99" → Must use in-memory data structures + aggressive caching

---

### 3. **CONSISTENCY** ← `accuracy` (CAP Theorem)
**System Design Interview Question:**
"Do you need strong consistency or eventual consistency?"

**Key Architectural Patterns:**
- **Strong Consistency**: 
  - Leader-based replication (single master)
  - Synchronous replication to followers
  - Linearizability guarantees
  
- **Eventual Consistency**:
  - Multi-master replication
  - Async replication
  - Conflict resolution strategy needed

**Trade-offs (CAP Theorem):**
- Consistency + Availability = Partition Intolerant
- Consistency + Partition Tolerance = Lower availability (CA is impossible)
- Availability + Partition Tolerance = Eventual Consistency

**Examples:**
- Easy: "95%+ accuracy, single region" → Can use strong consistency
- Medium: "99.5%+ accuracy, multi-region" → Must coordinate (strong with slower writes)
- Hard: "99.9%+ accuracy, global distribution" → Bounded eventual consistency needed

---

### 4. **EVENTUAL CONSISTENCY BOUNDS** ← `update_frequency`
**System Design Interview Question:**
"How stale can data be before it impacts user experience?"

**Key Architectural Patterns:**
- Replication lag bounds (e.g., <1 second)
- Conflict-free replicated data types (CRDTs)
- Vector clocks for causality
- Event sourcing for audit trails

**Trade-offs:**
- Tighter bounds = more replication traffic
- Looser bounds = better performance but stale data issues

**Examples:**
- Easy: "Hourly updates" → Can batch process, asynchronous updates fine
- Medium: "Streaming" → Need real-time propagation with <1s lag
- Hard: "Continuous" → Must handle partial updates, need causality tracking

---

### 5. **AVAILABILITY** ← `fault_tolerance`
**System Design Interview Question:**
"What's your SLA? How do you handle server/region failure?"

**Key Architectural Patterns:**
- **Low availability**: Single replica, periodic backups
- **Medium availability**: 2x replication, health checks
- **High availability**: 3x+ replication, automatic failover, circuit breakers
- **Critical availability (99.99% SLA)**: Full redundancy, chaos engineering

**Trade-offs:**
- More copies = higher cost, more replication traffic
- Faster failover = more complex orchestration
- Automated failover = risk of cascading failures

**Examples:**
- Easy: "Single-region, best effort" → Simple failover
- Medium: "Multi-region" → Need region-level failover
- Hard: "99.99% SLA" → Need automatic recovery, chaos testing

---

### 6. **LOAD BALANCING** ← `traffic_pattern`
**System Design Interview Question:**
"How do you handle traffic spikes? What's your scaling strategy?"

**Key Architectural Patterns:**
- **Steady state**: Round-robin, least-connections
- **Bursty traffic**: Auto-scaling, queue-based backpressure
- **Temporal spikes**: Predictive scaling, time-based provisioning
- **Geographic patterns**: Geo-routing, regional load balancing

**Trade-offs:**
- Aggressive auto-scaling = higher cost
- Conservative scaling = request timeouts during spikes
- Complex logic = harder to debug

**Examples:**
- Easy: "Steady requests" → Simple round-robin works
- Medium: "Bursty (peak hours)" → Auto-scaling group, request queue
- Hard: "Temporal spikes (scheduled times)" → Predictive scaling + queue

---

### 7. **CDN & GEOGRAPHIC DISTRIBUTION** ← `geography`
**System Design Interview Question:**
"How do you serve users globally with low latency?"

**Key Architectural Patterns:**
- **Single-region**: All users connect to one datacenter
- **Multi-region**: Content replicated across regions, geo-routing
- **Global CDN**: Static content on edge, compute in cloud

**Trade-offs:**
- Single region = simple but high latency for far users
- Multi-region = complex replication, consistency challenges
- CDN = expensive but much faster for users

**Examples:**
- Easy: "Single-region" → Simple: all traffic to one datacenter
- Medium: "Multi-region" → Replicate across 3-5 regions, geo-routing
- Hard: "Global" → CDN for edge delivery, distributed compute nodes

---

### 8. **MESSAGE QUEUES** ← `queueing_needs`
**System Design Interview Question:**
"How do you decouple components? Handle async processing?"

**Key Architectural Patterns:**
- **No queues**: Synchronous call chains (simple, monolithic)
- **Light queues**: Basic async for non-critical tasks
- **Heavy queues**: Event-driven architecture, CQRS, event sourcing

**Trade-offs:**
- Without queues: Simple but tight coupling, hard to scale
- With queues: Complex but decoupled, easier to handle spikes
- Heavy queuing: Powerful but operational complexity

**Examples:**
- Easy: "No queuing" → Synchronous processing, everything inline
- Medium: "Async fraud detection pipeline" → Heavy Kafka usage for event streams
- Hard: "Complex event processing" → Event sourcing + CQRS

---

### 9. **RATE LIMITING** ← `rate_limiting`
**System Design Interview Question:**
"How do you protect your API from abuse?"

**Key Architectural Patterns:**
- **No limiting**: Wide open (only for internal systems)
- **Per-user**: Token bucket per user ID
- **Per-IP**: Basic DDoS protection
- **Adaptive**: Priority-based, surge pricing, dynamic limits

**Trade-offs:**
- Simple limiting = unfair to bursty legitimate users
- Complex limiting = overhead, harder to debug
- No limiting = vulnerable to attacks

**Examples:**
- Easy: "No rate limiting needed" → Internal use only
- Medium: "Per-user rate limits" → Token bucket per user_id
- Hard: "Adaptive with priority" → Priority queue + dynamic rate adjustment

---

### 10. **SYSTEM DESIGN TRADEOFFS** ← `budget`
**System Design Interview Question:**
"Which tradeoffs would you make given budget constraints?"

**Key Architectural Patterns:**
- **Low budget**: 
  - Use managed services (AWS RDS instead of self-hosted)
  - Cache heavily to reduce DB load
  - Monolithic is simpler than microservices
  
- **Medium budget**:
  - Can afford some redundancy and multi-region
  - Dedicated infrastructure where it matters
  
- **High budget**:
  - Can implement all the patterns
  - Heavy investment in reliability/observability

**Trade-offs:**
- Low cost = constraint-driven architecture
- High cost = freedom but risk of overengineering
- Mid budget = must optimize carefully

---

## Task Difficulty Mapping

### EASY - Single Region, Straightforward
**Constraints:**
- Single use case (recommendation ranking)
- 100ms acceptable latency
- High (95%+) accuracy acceptable
- Moderate data (100GB-1TB)
- Hourly updates
- Eventual consistency OK
- Steady traffic
- Single region

**System Design Patterns:**
- Simple SQL database
- In-memory cache (Redis) for hot data
- Basic replication (read replicas)
- No queuingneeded

**Interview Focus:**
- Database design
- Caching strategy
- Basic indexing

---

### MEDIUM - Multi-Region, Trade-offs
**Constraints:**
- Fraud detection (critical)
- 500ms latency acceptable
- 99.5%+ accuracy required
- Large data (10TB+)
- Streaming updates
- Strong consistency needed (for fraud)
- Bursty traffic patterns
- Multi-region deployment

**System Design Patterns:**
- Large-scale database (sharded)
- Real-time event processing (Kafka)
- Strong consistency with multi-region replication
- Circuit breakers and fallbacks
- Auto-scaling for traffic

**Interview Focus:**
- CAP theorem (consistency required)
- Event streaming architecture
- Multi-region replication strategy
- Load balancing under bursty traffic

---

### HARD - Global Scale, Multiple Tradeoffs
**Constraints:**
- Multimodal assistant (complex)
- <50ms p99 latency (extreme)
- 99.9%+ accuracy (very high)
- Very large data (100TB+ models)
- Continuous updates (models change)
- Bounded eventual consistency
- Temporal spike patterns
- Global CDN distribution
- Critical 99.99% SLA
- Heavy event processing
- Adaptive rate limiting

**System Design Patterns:**
- Distributed model serving
- Event sourcing for audit
- CQRS for complex updates
- Global CDN + regional edge
- Advanced load balancing (latency-aware)
- Chaos engineering for reliability
- Priority queue scheduling

**Interview Focus:**
- CAP theorem tradeoffs in practice
- Distributed systems design
- Handling multiple conflicting constraints
- Global scalability patterns
- Chaos engineering for resilience

---

## Common Interview Questions You Can Now Answer

1. **"What's the CAP theorem and how does it apply here?"**
   - Accuracy constraint → Consistency requirement
   - Multi-region → Partition tolerance
   - You must choose: strong consistency OR availability

2. **"How would you design for 100x traffic growth?"**
   - Data size → Sharding strategy
   - Traffic pattern → Auto-scaling approach
   - Geography → Regional replication

3. **"What if a region goes down?"**
   - Fault tolerance → Failover strategy
   - Geography → Which regions are affected
   - QueuedNeeds → How to degrade gracefully

4. **"How do you maintain data consistency across regions?"**
   - Update frequency → Replication lag tolerance
   - Consistency requirement → Sync or async replication
   - Tradeoffs → Availability vs consistency

5. **"How would you handle this with 1/10th the budget?"**
   - Budget constraint → Remove redundancy
   - Scalability → Use managed services
   - Availability → Accept higher downtime

---

## System Design Simulator Verification

This environment now validates agent understanding of:

- ✅ CAP Theorem (consistency vs availability)
- ✅ Load balancing strategies
- ✅ Distributed systems (replication, sharding)
- ✅ API protection (rate limiting)
- ✅ Availability (fault tolerance, SLAs)
- ✅ Scalability (data volume, replication)
- ✅ Geographic distribution (CDN, multi-region)
- ✅ Async processing (message queues, CQRS)
- ✅ Tradeoff analysis (when to accept constraints)

**Score reflects understanding of:**
- How many constraints agent discovered (exploration completeness)
- Whether agent recognizes system design implications (concept mapping)
- Quality of final architectural recommendation (utilization of constraints)
- How well agent explains tradeoff reasoning (decision justification)
