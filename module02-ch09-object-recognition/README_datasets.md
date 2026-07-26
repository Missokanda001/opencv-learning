# Object Recognition Datasets Guide

Two dataset options for expanding your object recognition project (bag / dress / shoes).

---

## Option 1: Fashion-MNIST (Small & Fast)

**Best for**: Quick testing, classifier logic validation
**Size**: ~30MB
**Images**: 70,000 grayscale 28×28 images
**Classes**: 10 total — includes `Dress`, `Bag`, `Sneaker`, `Ankle boot`, `Sandal`



### Class labels
| Index | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | **Dress** |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | **Bag** |
| 9 | Ankle boot |

### ⚠️ Limitation
28×28 grayscale images are **too small for SIFT / Dense keypoint extraction**.
Use this only for testing your classifier pipeline (SVM, etc.), not for the BoVW feature approach.

---

## Option 2: Clothing Dataset Small (Recommended for your project)

**Best for**: SIFT / BoVW object recognition (matches your current code)
**Size**: ~300MB
**Images**: RGB real-world clothing photos
**Classes**: Dress, shoes, bags, and more


---

## Quick Comparison

| | Fashion-MNIST | Clothing Dataset Small |
|---|---|---|
| Download size | ~30MB | ~300MB |
| Image type | Grayscale 28×28 | RGB real photos |
| Good for SIFT/BoVW | ❌ No | ✅ Yes |
| Setup difficulty | 1 line of Python | Kaggle account + manual download |
| Realistic for your project | Limited | Best match |

---

