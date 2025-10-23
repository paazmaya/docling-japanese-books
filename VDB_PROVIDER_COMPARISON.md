# Vector Database Provider Comparison

This document compares vector database providers for the **docling-japanese-books** project, focusing on storage requirements, costs, and capabilities for Japanese document processing.

## 📊 Storage Requirements Analysis

### Project Configuration

- **Embedding Model**: BAAI/bge-m3 (1024 dimensions)
- **Target Content**: 80-page Japanese books
- **Chunking Strategy**: Late Chunking with ~400 characters per chunk
- **Overlap**: 10% between chunks for context preservation

### Storage Calculations per Book

| Content Type        | Chars/Page | Chunks/Book | Storage/Book | Books in 5GB |
| ------------------- | ---------- | ----------- | ------------ | ------------ |
| **Dense Classical** | 600        | 132 chunks  | 0.6 MB       | ~7,900       |
| **Mixed Content**   | 450        | 99 chunks   | 0.5 MB       | ~10,600      |
| **Modern Layout**   | 300        | 66 chunks   | 0.3 MB       | ~15,900      |

**Storage Breakdown per Chunk:**

- Vector embedding: 4KB (1024 dimensions × 4 bytes float32)
- Metadata: ~1KB (text content + document info)
- **Total per chunk**: ~5KB

## 🏢 Vector Database Provider Comparison

### 1. Zilliz Cloud (Recommended)

> **Sources**: [Zilliz Cloud Pricing](https://docs.zilliz.com/docs/pricing), [Zilliz Free Trials](https://docs.zilliz.com/docs/free-trials)

**Free Tier:**

- ✅ **Storage**: 5GB free
- ✅ **Capacity**: 7,000-15,000 books (80 pages each)
- ✅ **Features**: Full Milvus compatibility, managed service
- ✅ **Performance**: Enterprise-grade infrastructure
- ✅ **Integration**: Direct Milvus client support

**Paid Tiers:**

- **Starter**: $0.10/GB/month + compute
- **Standard**: Volume discounts, dedicated resources
- **Enterprise**: Custom pricing, SLA guarantees

**Pros:**

- Native Milvus compatibility (no code changes needed)
- Excellent free tier for development/research
- Global CDN and edge locations
- Built-in backup and disaster recovery
- Advanced security features

**Cons:**

- Newer service (less market presence than some competitors)
- Pricing can scale up quickly for large datasets

### 2. Pinecone

**Free Tier (Starter Plan):**

- ✅ **Storage**: Up to 2GB free (AWS us-east-1 only)
- ✅ **Capacity**: ~4,000 books (80 pages each)
- ✅ **Write Units**: Up to 2M/month
- ✅ **Read Units**: Up to 1M/month
- ✅ **Features**: Serverless, Console metrics, Community support
- ⚠️ **Limitations**: Single region (us-east-1), up to 5 indexes, 100 namespaces

**Paid Tiers:**

- **Standard**: 3-week trial with $300 credit, then $50/month minimum + pay-as-you-go
- **Enterprise**: Custom pricing with enhanced support and compliance features

**Current Pricing (Oct 2025):**

- Write Units: $0.4/1M writes
- Read Units: $0.5/1M reads
- Storage: Varies by region (~$0.096-0.128/GB/month)
- Additional features: SAML SSO, API key RBAC, object storage import

**Pros:**

- Mature platform with excellent performance
- Generous 2GB free tier (vs previous 100MB)
- Simple API and comprehensive documentation
- Built-in monitoring and metrics
- Strong ecosystem and integrations
- Multi-cloud availability (AWS, GCP, Azure)

**Cons:**

- Free tier limited to single region (us-east-1)
- $50/month minimum after trial period
- Can become expensive for high-volume applications
- Vendor lock-in (proprietary API)

### 3. Weaviate Cloud

> **Sources**: [Weaviate Pricing](https://weaviate.io/pricing), [Serverless Cloud](https://weaviate.io/deployment/serverless)

**Free Tier (Sandbox):**

- ⚠️ **Duration**: 14 days only (learning and prototyping)
- ❌ **Capacity**: Temporary use only, not for production
- ✅ **Features**: Full GraphQL API, hybrid search, monitoring
- ✅ **Support**: Community support via Public Slack

**Paid Tiers (Serverless Cloud):**

- **Standard**: $25/month base + $0.095 per 1M vector dimensions/month
- **Professional**: $135/month base + $0.145 per 1M vector dimensions/month
- **Business Critical**: $450/month base + $0.175 per 1M vector dimensions/month
- **Enterprise Cloud**: Custom pricing with dedicated resources

**Current Pricing (Oct 2025):**

For 10,000 books (1.02 billion vector dimensions):

- **Standard**: $122/month ($1,467/year)
- **Professional**: $283/month ($3,402/year)
- **Business Critical**: $629/month ($7,550/year)

**Pros:**

- Built-in hybrid search (BM25 + vector) out of the box
- GraphQL interface for complex queries
- Strong semantic capabilities and AI integrations
- Multiple SLA tiers with 24/7 support options
- Good open-source version available
- Quantization options (PQ, BQ) for cost optimization

**Cons:**

- No persistent free tier (14-day sandbox only)
- Expensive for large document collections ($122+/month minimum)
- More complex setup than Milvus/Pinecone
- Higher learning curve for GraphQL queries

### 4. Qdrant Cloud

> **Sources**: [Qdrant Pricing](https://qdrant.tech/pricing/), [Qdrant Calculator](https://cloud.qdrant.io/calculator)

**Free Tier:**

- ✅ **Storage**: 1GB free forever (no credit card required)
- ✅ **Capacity**: ~2,600 books (80 pages each)
- ✅ **Performance**: Rust-based, very fast
- ✅ **Features**: Full managed service, multiple cloud providers

**Paid Tiers:**

- **Hybrid Cloud**: $0.014/hour = $10.08/month base cluster + scaling costs
- **Private Cloud**: Custom pricing (enterprise)
- **Marketplace**: Available on AWS, GCP, and Azure marketplaces

**Current Pricing (Oct 2025):**

For 10,000 books (~4GB storage):

- **Base cluster**: $10.08/month (720 hours × $0.014)
- **Storage scaling**: ~$1.50/month (estimated 3GB overage × $0.50/GB)
- **Total estimated**: $11.58/month for 10K books

**Pros:**

- Excellent performance (Rust implementation)
- Best value free tier (1GB permanent vs competitors)
- Advanced filtering and search capabilities
- Open source with strong community
- Multi-cloud deployment options
- No vendor lock-in (can self-host)
- Built-in quantization and optimization features

**Cons:**

- Smaller managed cloud ecosystem compared to Pinecone/Weaviate
- Would require API migration from Milvus
- Scaling costs can vary significantly by configuration
- Less documentation for complex enterprise scenarios

### 5. Local Milvus Lite (Current Default)

**Costs:**

- ✅ **Storage**: Limited only by local disk space
- ✅ **Capacity**: Unlimited books (hardware dependent)
- ✅ **Features**: Full Milvus feature set

**Infrastructure Requirements:**

- **RAM**: 8GB+ recommended for large collections
- **Storage**: SSD recommended for performance
- **CPU**: Multi-core for embedding generation

**Pros:**

- Complete control and privacy
- No usage-based costs
- Full feature access
- Ideal for development and testing

**Cons:**

- No automatic scaling
- Manual backup and maintenance
- Single point of failure
- No built-in redundancy

## 🎯 Recommendations by Use Case

### 📚 Research/Academic Projects (5,000-15,000 books)

**Recommendation: Zilliz Cloud Free Tier**

- ✅ 5GB storage handles most research collections
- ✅ No cost for development and small-scale research
- ✅ Easy migration to paid tiers if needed
- ✅ Full Milvus compatibility

**Alternatives**:

- **Qdrant Cloud**: 1GB free tier for smaller collections (~2,600 books)
- **Weaviate**: 14-day sandbox for hybrid search experimentation

### 🏢 Production Applications (>15,000 books)

**Recommendation: Zilliz Cloud Paid or Pinecone Standard**

- **Zilliz Cloud**: Better for Milvus compatibility, slightly lower cost
- **Pinecone**: Better for performance, mature ecosystem, $50-75/month range
- Local Milvus Lite for development and testing
- Consider data partitioning strategies

### 💼 Small-Medium Production (5,000-15,000 books)

**Recommendation: Qdrant Cloud or Zilliz Cloud**

- **Qdrant**: $10-12/month, excellent Rust performance, cost-effective scaling
- **Zilliz Cloud**: Free tier possible, better for budget-conscious projects
- **Pinecone**: $50/month minimum, excellent performance, proven reliability
- All offer enterprise-grade reliability and support

### 💻 Development/Testing

**Recommendation: Local Milvus Lite**

- ✅ Zero cost for development
- ✅ Full feature compatibility
- ✅ Fast iteration and testing
- Easy migration to cloud when ready

### 🔬 Experimental/Prototyping

**Recommendation: Multiple Providers**

- Start with Local Milvus Lite for development
- Test with Zilliz Cloud free tier
- Evaluate others based on specific requirements

## 🚀 Migration Strategy

### Phase 1: Development (Current)

```bash
# Local development with Milvus Lite
uv run docling-japanese-books config-db --mode local
```

### Phase 2: Cloud Testing

```bash
# Test with Zilliz Cloud free tier
export ZILLIZ_CLOUD_URI="your-cluster-uri"
export ZILLIZ_API_KEY="your-api-key"
uv run docling-japanese-books config-db --mode cloud --test-connection
```

### Phase 3: Production Scaling

- Monitor usage and performance
- Implement data partitioning if needed
- Consider hybrid deployment for optimal costs

## 📈 Cost Projections

### Storage Costs (per 10,000 books ≈ 5GB)

> **Sources**: [Zilliz Cloud Pricing](https://docs.zilliz.com/docs/pricing), [Pinecone Pricing](https://www.pinecone.io/pricing/), [Weaviate Pricing](https://weaviate.io/pricing)

| Provider         | Monthly Cost    | Annual Cost | Notes                           |
| ---------------- | --------------- | ----------- | ------------------------------- |
| **Zilliz Cloud** | $0 (free) → $50 | $0 → $600   | 5GB free tier → Standard        |
| **Qdrant**       | $0 → $12        | $0 → $144   | 1GB free tier, $10/month base   |
| **Pinecone**     | $50-75          | $600-900    | 2GB free, $50/month minimum     |
| **Weaviate**     | $122+           | $1,467+     | No free tier, $25/month minimum |
| **Local Milvus** | $20-50          | $240-600    | Hardware/hosting only           |

### Performance Considerations

| Provider         | Latency  | Throughput | Scalability | Maintenance |
| ---------------- | -------- | ---------- | ----------- | ----------- |
| **Zilliz Cloud** | Low      | High       | Excellent   | None        |
| **Pinecone**     | Very Low | Very High  | Excellent   | None        |
| **Weaviate**     | Low      | High       | Good        | None        |
| **Qdrant**       | Very Low | Very High  | Good        | None        |
| **Local Milvus** | Very Low | Medium     | Manual      | High        |

## 🔧 Implementation Notes

### Current Project Support

The **docling-japanese-books** project currently supports:

- ✅ Local Milvus Lite
- ✅ Zilliz Cloud
- ❌ Other providers (would require API adapters)

### Adding New Providers

To add support for other providers:

1. Create provider-specific client adapters
2. Implement common interface in `vector_db.py`
3. Add configuration options in `config.py`
4. Update CLI commands for provider selection

### Configuration Examples

```python
# Zilliz Cloud
config = DatabaseConfig(
    deployment_mode='cloud',
    zilliz_cloud_uri='https://your-cluster.cloud.zilliz.com',
    zilliz_api_key='your-api-key'
)

# Local Milvus
config = DatabaseConfig(
    deployment_mode='local',
    database_path='./milvus_data'
)
```

## 📋 Conclusion

For the **docling-japanese-books** project processing 80-page Japanese books:

1. **Best Free Option**: Zilliz Cloud (5GB = ~10,000 books) vs Pinecone (2GB = ~4,000 books)
2. **Best Development**: Local Milvus Lite (unlimited, no cloud provider costs)
3. **Best Production (Small-Medium)**: Pinecone Standard ($50-75/month, proven reliability)
4. **Best Production (Large Scale)**: Zilliz Cloud paid tiers (better cost efficiency)
5. **Best Performance**: Pinecone or local setup with SSD storage

### Cost-Performance Analysis:

| Provider     | 10K Books Cost | Free Tier     | Best For                  |
| ------------ | -------------- | ------------- | ------------------------- |
| Zilliz Cloud | $0-50/month    | 5GB permanent | Budget projects, research |
| Qdrant       | $0-12/month    | 1GB permanent | Cost-effective production |
| Pinecone     | $50-75/month   | 2GB permanent | Production reliability    |
| Weaviate     | $122+/month    | 14-day only   | Advanced hybrid search    |

The current dual-mode implementation (local + Zilliz Cloud) provides the optimal balance of development flexibility and production scalability, with Pinecone as a competitive alternative and Weaviate for specialized hybrid search needs.

## 📚 Sources & Validation

All pricing information was gathered from official sources in October 2025:

### Primary Pricing Sources:

- **Zilliz Cloud**: [https://docs.zilliz.com/docs/pricing](https://docs.zilliz.com/docs/pricing)
- **Zilliz Free Trials**: [https://docs.zilliz.com/docs/free-trials](https://docs.zilliz.com/docs/free-trials)
- **Qdrant**: [https://qdrant.tech/pricing/](https://qdrant.tech/pricing/)
- **Qdrant Calculator**: [https://cloud.qdrant.io/calculator](https://cloud.qdrant.io/calculator)
- **Pinecone**: [https://www.pinecone.io/pricing/](https://www.pinecone.io/pricing/)
- **Weaviate**: [https://weaviate.io/pricing](https://weaviate.io/pricing)
- **Weaviate Serverless**: [https://weaviate.io/deployment/serverless](https://weaviate.io/deployment/serverless)

### Documentation Sources:

- **Weaviate Cloud Docs**: [https://docs.weaviate.io/cloud](https://docs.weaviate.io/cloud)
- **Pinecone Documentation**: [https://docs.pinecone.io/](https://docs.pinecone.io/)
- **Pinecone Billing Guide**: [https://docs.pinecone.io/guides/organizations/manage-billing/understand-billing](https://docs.pinecone.io/guides/organizations/manage-billing/understand-billing)
- **Milvus Documentation**: [https://milvus.io/docs](https://milvus.io/docs)

### Methodology & Analysis Process:

This comparison was conducted through systematic analysis:

1. **Data Gathering**: Official pricing pages scraped and analyzed (October 2025)
2. **Storage Calculations**: Based on BGE-M3 embeddings and Japanese text characteristics
3. **Cost Modeling**: Calculated realistic usage scenarios for 5GB-10GB document collections
4. **Validation**: Cross-referenced with multiple official sources and documentation

### Calculation Assumptions:

- **BGE-M3 embeddings**: 1024 dimensions per vector
- **Japanese books**: 80 pages, ~100 chunks per book (400-500 characters per page)
- **Storage per chunk**: ~5KB (4KB vector + 1KB metadata)
- **Usage patterns**: Moderate query volumes (1000-5000 queries/day)
- **Chunking strategy**: Late Chunking with 10% overlap for context preservation

### Price Analysis Scenarios:

- **Zilliz Cloud**: 5GB free tier capacity analysis
- **Pinecone**: 2GB free tier + Standard plan calculations ($50/month minimum)
- **Weaviate**: Serverless Cloud dimensions-based pricing ($25/month base + $0.095/1M dims)
- **Local Milvus**: Hardware and hosting cost estimates

**Note**: Pricing subject to change. Verify current rates on official provider websites before making decisions. This analysis reflects October 2025 pricing and should be updated periodically.
