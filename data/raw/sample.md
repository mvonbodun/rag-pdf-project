# Introduction to Distributed Systems

Distributed systems consist of multiple independent computers that appear to the user as a single coherent system. They enable scalability and fault tolerance but introduce challenges such as consensus and partial failure.

## Scalability and Availability

Horizontal scalability allows adding more nodes to improve throughput. Availability measures the proportion of time a system is operational. Replication increases availability but may complicate consistency guarantees.

## Consistency Models

CAP theorem states that in the presence of a network partition, a system must choose between consistency and availability. Strong consistency ensures that all clients see the same data after an update. Eventual consistency allows temporary divergence that converges over time.

## Consensus

Consensus protocols like Paxos and Raft enable a cluster to agree on a single value despite failures. Raft focuses on understandability with concepts like leaders, terms, and logs. These protocols tolerate crash faults under quorum assumptions.

## Fault Tolerance

Redundancy and failure detection are core. Heartbeats and timeouts are used to suspect failure, but false positives can occur under high latency. Idempotent operations simplify retries in the presence of partial failures.
