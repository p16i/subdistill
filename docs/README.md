# Knowledge Distillation Documentation

Welcome to the knowledge distillation frameworks documentation. This repository implements 7 state-of-the-art knowledge distillation frameworks with comprehensive analysis and comparison.

## 📚 Documentation Structure

### 🚀 [Quick Reference Guide](kd_frameworks_quick_reference.md)
**Best for**: Getting started, framework selection, implementation tips
- Framework selection matrix
- Performance vs complexity chart  
- Common pitfalls and solutions
- Implementation code snippets

### 📄 [One-Page Summary](kd_frameworks_summary.md)
**Best for**: Quick decisions, overview, recommendations
- Decision tree for framework selection
- Traffic light recommendation system
- Essential metrics comparison
- Learning roadmap

### 📊 [Comprehensive Comparison](knowledge_distillation_frameworks_comparison.md)
**Best for**: Detailed analysis, research, understanding trade-offs
- Complete framework descriptions
- Detailed pros and cons analysis
- Use case recommendations
- Comparison matrix with all metrics

### 🔬 [Technical Analysis](kd_frameworks_technical_analysis.md)
**Best for**: Research, production deployment, technical decision making
- Computational complexity analysis
- Empirical performance results
- Implementation considerations
- Future research directions

## 🎯 Quick Navigation

### I want to...

**Make a quick decision** → [One-Page Summary](kd_frameworks_summary.md)

**Get started quickly** → [Quick Reference](kd_frameworks_quick_reference.md)

**Choose the right framework** → [Framework Selection Guide](kd_frameworks_quick_reference.md#framework-selection-guide)

**Understand the details** → [Comprehensive Comparison](knowledge_distillation_frameworks_comparison.md)

**Make technical decisions** → [Technical Analysis](kd_frameworks_technical_analysis.md)

**See performance numbers** → [Technical Analysis - Performance](kd_frameworks_technical_analysis.md#performance-analysis)

**Avoid common mistakes** → [Quick Reference - Pitfalls](kd_frameworks_quick_reference.md#common-pitfalls)

## 🏗️ Implemented Frameworks

| Framework | Year | Type | Complexity | Performance |
|-----------|------|------|------------|-------------|
| **KD** | 2015 | Output-level | Low | Good |
| **FitNet** | 2015 | Feature-level | Medium | Good |
| **Attention Transfer** | 2017 | Attention-based | Low | Good |
| **VID** | 2019 | Feature-level | High | Good |
| **SPKD** | 2019 | Relation-based | Medium | Good |
| **DKD** | 2021 | Output-level | Low | Very Good |
| **VkD** | 2024 | Feature-level | Medium | Excellent |

## 🔍 Framework Categories

- **Output-level**: KD, DKD - Work on final predictions
- **Feature-level**: FitNet, VID, VkD - Transfer intermediate representations  
- **Attention-based**: Attention Transfer - Focus on attention mechanisms
- **Relation-based**: SPKD - Preserve sample relationships

## 📈 Usage Recommendations

### For Beginners
Start with [Quick Reference](kd_frameworks_quick_reference.md) → Try **KD** framework

### For Production
Read [Technical Analysis](kd_frameworks_technical_analysis.md) → Use **KD** or **DKD**

### For Research  
Study [Comprehensive Comparison](knowledge_distillation_frameworks_comparison.md) → Explore **VkD** or **VID**

### For Computer Vision
Check [Framework Selection](kd_frameworks_quick_reference.md#use-case-matrix) → Consider **Attention Transfer** or **VkD**

---

*Last updated: 2024 | Repository: [p16i/xai-kd](https://github.com/p16i/xai-kd)*