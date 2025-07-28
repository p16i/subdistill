# Knowledge Distillation Frameworks Summary

## One-Page Framework Comparison

### 🎯 Framework Selection Decision Tree

```
START: What do you need?

├── Simple baseline & understanding
│   └── 📝 KD (Hinton 2015)
│
├── Best classification performance  
│   └── 🚀 DKD (Zhao 2021)
│
├── State-of-the-art results
│   └── 🔥 VkD (Miles 2024)
│
├── Different teacher-student architectures
│   └── 🔧 FitNet (Romero 2015)
│
├── Spatial attention (Computer Vision)
│   └── 👁️ Attention Transfer (Zagoruyko 2017)
│
├── Theoretical guarantees
│   └── 📐 VID (Ahn 2019)
│
└── Sample relationship preservation
    └── 🔗 SPKD (Tung 2019)
```

### ⚡ Performance Ranking

```
🥇 VkD (2024)        - Excellent performance, modern approach
🥈 DKD (2021)        - Very good for classification, simple
🥉 VID (2019)        - Good with theoretical backing
4️⃣ FitNet (2015)     - Good for architecture transfer
5️⃣ SPKD (2019)       - Good for relationships
6️⃣ KD (2015)         - Good baseline, universally applicable
7️⃣ Attention (2017)  - Good for spatial tasks
```

### 🛠️ Implementation Difficulty

```
Easy    ⭐⭐⭐     KD, DKD, Attention Transfer
Medium  ⭐⭐       FitNet, SPKD, VkD
Hard    ⭐         VID
```

### 💰 Computational Cost

```
Low     💚💚💚   KD, DKD, Attention Transfer  
Medium  💛💛     FitNet, VkD
High    ❤️       VID, SPKD (large batches)
```

### 📊 Essential Metrics Comparison

| Framework | Performance | Simplicity | Speed | Versatility | Year |
|-----------|-------------|------------|-------|-------------|------|
| **KD** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2015 |
| **FitNet** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2015 |
| **Attention** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 2017 |
| **VID** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 2019 |
| **SPKD** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 2019 |
| **DKD** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 2021 |
| **VkD** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2024 |

### 🚦 Traffic Light System

| Framework | Recommendation |
|-----------|----------------|
| **KD** | 🟢 Always good choice - start here |
| **DKD** | 🟢 Excellent for classification |
| **VkD** | 🟢 Best performance if complexity OK |
| **FitNet** | 🟡 Good for architectural differences |
| **Attention** | 🟡 Good for computer vision |
| **SPKD** | 🟡 Good for specific relationship tasks |
| **VID** | 🔴 Use only if theoretical rigor required |

### 📈 Learning Curve

```
Week 1: KD           (Learn fundamentals)
Week 2: DKD          (Improve performance)
Week 3: VkD/FitNet   (Advanced techniques)
Week 4: VID/SPKD     (Specialized approaches)
```

### 🎯 Final Recommendations

**🔥 Top 3 for Most Users:**
1. **DKD** - Best balance of performance and simplicity
2. **KD** - Perfect learning baseline
3. **VkD** - Cutting-edge performance

**⚠️ Use With Caution:**
- **VID** - Complex, use only for research
- **SPKD** - Memory intensive with large batches

**💡 Pro Tips:**
- Start with KD to understand concepts
- Move to DKD for better classification
- Try VkD when you need best performance
- Use FitNet for very different architectures

---

**Need more details?** See our [complete documentation](README.md)!