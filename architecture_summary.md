# Siamese Neural Network Architecture Summary

This document provides a detailed breakdown of the Siamese Neural Network architecture used for Face Verification / Recognition.

---

## 1. Network Overview & Pipeline Flow

The network compares two input images simultaneously using twin subnetworks that share identical weights.

```
Anchor Image (105x105x3) ────► ┌───────────────────────────────────┐
                               │  Shared Base Embedding Network    │ ──► Vector E1 (4096,) ┐
                               │  (make_embedding)                 │                       │
Validation Image (105x105x3) ─► └───────────────────────────────────┘ ──► Vector E2 (4096,) ┴─► L1Dist Layer ──► Distance (4096,) ──► Dense(1, Sigmoid) ──► Match Score (0 to 1)
                                 ▲ Shared Weights for both inputs
```

---

## 2. Stage-by-Stage Layer Breakdown

### Stage 1: Dual Input Layer
input_img: Tensor of shape (None, 105, 105, 3) expecting the Anchor face image.
validation_img: Tensor of shape (None, 105, 105, 3) expecting the comparison face image.

---

### Stage 2: Shared Embedding Network (`make_embedding`)
Both input images are passed through the exact same feature extraction pipeline:

| Layer / Block                   | Output Shape            | Parameters  | Purpose                                         |
| :---                            | :---                    | :---        | :---                                            |
| **Input**                       | `(None, 105, 105, 3)`   | 0           | Raw RGB Image Input                             |
| **`Conv2D(64, 10x10, relu)`**   | `(None, 96, 96, 64)`    | 19,264      | Extracts 64 low-level edge & line feature maps  |
| **`MaxPooling2D(2x2)`**         | `(None, 48, 48, 64)`    | 0           | Downsamples height & width by 2x                |
| **`Conv2D(128, 7x7, relu)`**    | `(None, 42, 42, 128)`   | 401,536     | Extracts 128 mid-level texture & curve patterns |
| **`MaxPooling2D(2x2)`**         | `(None, 21, 21, 128)`   | 0           | Downsamples height & width by 2x                |
| **`Conv2D(128, 4x4, relu)`**    | `(None, 18, 18, 128)`   | 262,272     | Extracts 128 high-level facial shapes           |
| **`MaxPooling2D(2x2)`**         | `(None, 9, 9, 128)`     | 0           | Downsamples height & width by 2x                |
| **`Conv2D(256, 4x4, relu)`**    | `(None, 6, 6, 256)`     | 524,544     | Extracts 256 high-level facial identity traits  |
| **`Flatten()`**                 | `(None, 9216)`          | 0           | Unrolls (6x6x256) feature grid to 1D vector     |
| **`Dense(4096, Sigmoid)`**      | `(None, 4096)`          | 37,752,832  | Compresses features to normalized 4096-dim face |

---

### Stage 3: Distance Calculation & Binary Classification

**L1Dist Layer**: Distance = E_anchor - E_validation

Calculates the element-wise absolute difference between the 4,096 values of the twin face embeddings.
If faces match  -> Distance values are close to 0.0.
If faces differ -> Distance values are significantly positive.

**Dense(1, Sigmoid) Layer:**

Takes the 4,096 distance values and weighs them down into a single output number.
Sigmoid converts the output into a probability percentage:
Near 1.0: Same Person (Match)
Near 0.0: Different People (No Match)
