## Message to Team: Why We're Using PostgreSQL+pgvector Instead of Pure Spark

Hey team,

I wanted to clarify our tech stack choice, specifically **why we're using PostgreSQL+pgvector for the search layer instead of a pure Spark solution** (even though HNSWlib-Spark exists).

### TL;DR
Spark is great for **batch ETL**, but **PostgreSQL+pgvector is purpose-built for real-time vector search** in production. We use both: Spark for offline processing, PostgreSQL for online queries.

---

### Why Not Pure Spark?

**1. Spark Is Batch-Oriented, Not Query-Oriented**
- Spark excels at **processing large datasets** (our ETL pipeline)
- It's **not designed for low-latency, concurrent user queries**
- Starting a Spark job for each search query would take **seconds**, not milliseconds
- PostgreSQL responds in **50-200ms** vs Spark's **2-10 seconds** startup overhead

**2. HNSWlib-Spark Limitations**
While HNSWlib-Spark integrates HNSW with Spark MLlib, it has critical issues:
- **No persistent index**: Needs to rebuild HNSW index on every Spark session restart
- **No concurrent query support**: Spark isn't built for multi-user real-time access
- **Resource inefficient**: Requires keeping Spark cluster running 24/7 just for queries
- **Latency**: Even with HNSW, Spark's RDD/DataFrame overhead adds 100-500ms per query

**3. Production Requirements**
Our course project simulates a **real production system** where users expect:
- **Instant search results** (<200ms)
- **Concurrent access** (multiple users searching simultaneously)
- **Always-on availability** (no cold starts)
- **Cost efficiency** (don't need a Spark cluster running 24/7)

---

### Our Hybrid Architecture (Best of Both Worlds)

```
[Offline - Spark] 
ETL + Data Processing (batch, runs once)
    ↓
[Online - PostgreSQL+pgvector]
Real-time Search (low latency, always available)
```

**Spark's Role** (Where it excels):
- Process 900MB MovieLens ratings.csv
- Merge datasets from multiple sources
- Filter and aggregate data
- Generate Movie Soup text

**PostgreSQL+pgvector's Role** (Where Spark struggles):
- Store 15,000 movie embeddings with HNSW index
- Handle real-time user queries in <200ms
- Support concurrent searches from multiple users
- Provide persistent, always-on service

---

### Comparison Table

| Feature | Pure Spark Solution | Our Hybrid (Spark+PostgreSQL) |
|---------|-------------------|-------------------------------|
| **ETL Performance** | ✅ Excellent | ✅ Excellent (same) |
| **Query Latency** | ❌ 2-10 seconds | ✅ 50-200ms |
| **Concurrent Users** | ❌ Limited | ✅ Hundreds+ |
| **Startup Time** | ❌ Cold start every query | ✅ Always-on |
| **Resource Cost** | ❌ Cluster 24/7 | ✅ Small DB instance |
| **Index Persistence** | ❌ Rebuild on restart | ✅ Permanent |
| **Course Relevance** | ⚠️ Half the story | ✅ Full big data stack |

---

### Course Grading Perspective

Our approach actually demonstrates **MORE big data management concepts**:

**With Pure Spark:**
- ✅ Distributed processing
- ❌ Missing: Database indexing, query optimization, OLTP/OLAP hybrid workloads

**With Our Hybrid:**
- ✅ Distributed processing (Spark ETL)
- ✅ Database indexing (HNSW)
- ✅ Query optimization (vector similarity + SQL filters)
- ✅ Bulk data loading (COPY vs INSERT)
- ✅ Hot/cold data separation
- ✅ Production architecture design

---

### Bottom Line

Think of it like this:
- **Spark** = Factory (great for manufacturing/processing)
- **PostgreSQL** = Store (great for serving customers)

You wouldn't ask customers to wait in the factory while you process their order from scratch. Similarly, we use Spark to **prepare** the data once, then PostgreSQL to **serve** it efficiently to users.

---

Let me know if you have questions!

Best,
[Your Name]

---

**Technical References:**
- pgvector GitHub: https://github.com/pgvector/pgvector
- HNSWlib-Spark limitations: https://github.com/kno10/java-hnsw/issues/9
- HNSW Paper: https://arxiv.org/abs/1603.09320
