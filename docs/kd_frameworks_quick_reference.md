# Knowledge Distillation Frameworks - Quick Reference

## Framework Selection Guide

### 📊 Quick Comparison

| Need | Best Framework | Reason |
|------|---------------|--------|
| **Simple baseline** | KD (Hinton 2015) | Easy to implement, well-understood |
| **Best performance** | VkD (Miles 2024) | State-of-the-art results |
| **Classification improvement** | DKD (Zhao 2021) | Better than standard KD with minimal complexity |
| **Spatial attention** | Attention Transfer | Focus on important image regions |
| **Different architectures** | FitNet | Handles architectural differences well |
| **Theoretical rigor** | VID | Variational information theory |
| **Sample relationships** | SPKD | Preserves semantic similarities |

### 🚀 Implementation Complexity

```
Simple     ████████████ KD, DKD, Attention Transfer
Medium     ████████     FitNet, SPKD, VkD  
Complex    ████         VID
```

### ⚡ Computational Cost (Training)

```
Low        ████████████ KD, DKD, Attention Transfer
Medium     ████████     FitNet, VkD
High       ████         VID, SPKD (large batches)
```

### 🎯 Use Case Matrix

| Task Type | Recommended Frameworks |
|-----------|----------------------|
| **Image Classification** | DKD → VkD → KD |
| **Object Detection** | Attention Transfer → FitNet → VkD |
| **Semantic Segmentation** | FitNet → VkD → Attention Transfer |
| **Metric Learning** | SPKD → VkD → VID |
| **Model Compression** | KD → DKD → VkD |
| **Architecture Transfer** | FitNet → VkD → VID |

### 📈 Performance vs Complexity

```
Performance
     ↑
     |    VkD ●
     |       
     |  DKD ●     ● VID
     |           
     | AT ●  ● SPKD
     | KD ●   ● FitNet
     |
     └────────────────→ Complexity
```

### 🔧 Implementation Tips

**KD (Hinton 2015)**
```python
# Key hyperparameter: temperature
temperature = 4  # Start here, tune between 1-10
```

**DKD (Zhao 2021)**
```python
# Balance target vs non-target knowledge
alpha = 1.0  # Target class knowledge
beta = 8.0   # Non-target class knowledge
```

**FitNet (Romero 2015)**
```python
# Layer selection is crucial
# Use middle layers, not too early/late
hint_layer = "layer3"  # For ResNet
```

**Attention Transfer**
```python
# P-norm selection affects attention quality
p_norm = 2  # L2 norm works well for most cases
```

### ⚠️ Common Pitfalls

1. **KD**: Temperature too high/low → poor knowledge transfer
2. **FitNet**: Wrong hint layer selection → degraded performance  
3. **VID**: Poor initialization → training instability
4. **SPKD**: Large batches → memory issues
5. **DKD**: Wrong alpha/beta ratio → imbalanced learning

### 📚 Paper References

- **KD**: Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
- **FitNet**: Romero et al. "FitNets: Hints for Thin Deep Nets" (2015) 
- **AT**: Zagoruyko & Komodakis "Paying More Attention to Attention" (2017)
- **VID**: Ahn et al. "Variational Information Distillation" (2019)
- **SPKD**: Tung & Mori "Similarity-Preserving Knowledge Distillation" (2019)
- **DKD**: Zhao et al. "Decoupled Knowledge Distillation" (2021)
- **VkD**: Miles et al. "VkD: Improving Knowledge Distillation using Orthogonal Projections" (2024)

### 🏃 Getting Started

1. **Beginners**: Start with `KD` for understanding concepts
2. **Quick wins**: Use `DKD` for better classification performance
3. **Research**: Explore `VkD` for state-of-the-art results
4. **Production**: Consider `KD` or `DKD` for reliability

For detailed analysis, see [Complete Comparison](knowledge_distillation_frameworks_comparison.md).