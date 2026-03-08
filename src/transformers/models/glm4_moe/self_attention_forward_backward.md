# Self-Attention Forward & Backward 수학적 유도

## 표기법

| 기호 | 의미 | Shape |
|------|------|-------|
| Q | Query 행렬 | (n, d) |
| K | Key 행렬 | (n, d) |
| V | Value 행렬 | (n, d) |
| S | Scaled dot-product score | (n, n) |
| P | Attention probability (softmax 출력) | (n, n) |
| O | Attention output | (n, d) |
| d | Head dimension | scalar |
| dX | ∂L/∂X (Loss에 대한 X의 gradient) | X와 동일 |

---

## Forward Pass

```
S = Q @ K.T / sqrt(d)    ← scaled dot-product
P = softmax(S)            ← row-wise softmax
O = P @ V                 ← weighted sum of values
```

---

## Backward Pass

역순으로 진행: `dO → dP, dV → dS → dQ, dK`

---

### 1. Matmul Backward 기본 공식

> `C = A @ B` 일 때:
> - `dA = dC @ B.T`
> - `dB = A.T @ dC`
>
> **규칙: 미분 대상이 아닌 쪽을 전치해서, dC와의 차원이 맞도록 곱한다.**

| 구하려는 것 | 전치하는 것 | 공식 | 차원 검증 |
|------------|-----------|------|----------|
| dA (m,k) | B를 전치 | dC @ B.T | (m,n) @ (n,k) = (m,k) ✓ |
| dB (k,n) | A를 전치 | A.T @ dC | (k,m) @ (m,n) = (k,n) ✓ |

**외울 필요 없이 차원만 맞춰보면 공식이 자동으로 유도된다.**

#### 숫자 예시

```
P = [[1, 2],    V = [[5, 6],
     [3, 4]]         [7, 8]]

O = P @ V = [[1*5+2*7, 1*6+2*8],   = [[19, 22],
             [3*5+4*7, 3*6+4*8]]      [43, 50]]
```

dO = I (단위행렬)로 놓으면:

```
dV = P.T @ dO = [[1, 3],  @ [[1, 0],  = [[1, 3],
                  [2, 4]]    [0, 1]]     [2, 4]]

dP = dO @ V.T = [[1, 0],  @ [[5, 7],  = [[5, 7],
                  [0, 1]]    [6, 8]]     [6, 8]]
```

**검증 (perturbation)**: V_11 = 5를 ε 흔들면:
- O_11에 P_11 = 1만큼, O_21에 P_21 = 3만큼 영향
- dL/dV_11 = dO_11 * 1 + dO_21 * 3 = 1*1 + 0*3 = 1 ✓

---

### 2. `O = P @ V`의 Backward

matmul 공식 직접 적용:

```
dP = dO @ V.T
dV = P.T @ dO
```

---

### 3. `P = softmax(S)`의 Backward

#### Softmax의 Jacobian

`P_i = exp(S_i) / Σ_j exp(S_j)` 이므로:

```
∂P_i/∂S_j = P_i * (δ_ij - P_j)
```

- i = j일 때: `P_i * (1 - P_i)` — 자기 자신에 대한 미분
- i ≠ j일 때: `-P_i * P_j` — 다른 원소에 대한 미분

#### Chain Rule 적용

```
dS_i = Σ_j dP_j * ∂P_j/∂S_i
     = Σ_j dP_j * P_j * (δ_ji - P_i)
     = dP_i * P_i  -  P_i * Σ_j(dP_j * P_j)
     = P_i * (dP_i  -  Σ_j dP_j * P_j)
```

행렬 표현 (row-wise):

```
dS = P * (dP - rowsum(dP * P))
```

여기서 `*`는 element-wise 곱, `rowsum(dP * P)`는 각 행의 `Σ_j dP_j * P_j` 스칼라를 broadcast한 것.

#### 숫자 예시

```
P = [0.2, 0.8],   dP = [1, -1]
```

| Step | 계산 | 결과 |
|------|------|------|
| dP * P (element-wise) | [1×0.2, (-1)×0.8] | [0.2, -0.8] |
| rowsum(dP * P) | 0.2 + (-0.8) | -0.6 |
| dP - rowsum | [1-(-0.6), -1-(-0.6)] | [1.6, -0.4] |
| P * (결과) | [0.2×1.6, 0.8×(-0.4)] | **[0.32, -0.32]** |

**검증**: S_1을 ε 흔들면:
- P_1 변화: P_1(1-P_1)ε = 0.2×0.8×ε = 0.16ε
- P_2 변화: -P_1×P_2×ε = -0.16ε
- dL = dP_1×0.16ε + dP_2×(-0.16ε) = 1×0.16ε + (-1)×(-0.16ε) = 0.32ε
- → dS_1 = 0.32 ✓

---

### 4. `S = S_raw / sqrt(d)`의 Backward

상수 나누기의 미분은 그대로 나누기:

```
dS_raw = dS / sqrt(d)
```

(이후 단계에서 dS를 dS/sqrt(d)로 덮어쓴다.)

---

### 5. `S_raw = Q @ K.T`의 Backward

matmul 공식 적용. A=Q, B=K.T로 놓으면:

```
dQ = dS @ (K.T).T = dS @ K
d(K.T) = Q.T @ dS  →  전치하면  →  dK = dS.T @ Q
```

#### 숫자 예시

```
Q = [[1, 0],    K = [[1, 1],    sqrt(d) = 1.41
     [0, 1]]         [0, 1]]
```

Forward:
```
S_raw = Q @ K.T = [[1, 0],  @ [[1, 0],  = [[1, 0],
                    [0, 1]]    [1, 1]]     [1, 1]]
```

Backward (dS_scaled = dS / sqrt(d)):
```
dQ = dS_scaled @ K
dK = dS_scaled.T @ Q
```

---

## 전체 Backward 요약

```python
# 이미 알고 있는 것: dO (Loss에서 역전파된 gradient)

# Step 1: O = P @ V
dP = dO @ V.T
dV = P.T @ dO

# Step 2: P = softmax(S)
dS = P * (dP - rowsum(dP * P))

# Step 3: S = Q @ K.T / sqrt(d)
dS = dS / sqrt(d)
dQ = dS @ K
dK = dS.T @ Q
```

---

## 핵심 정리

| 개념 | 공식 | 핵심 |
|------|------|------|
| Matmul backward | dA = dC @ B.T, dB = A.T @ dC | 미분 대상 아닌 쪽을 전치 |
| Softmax backward | P * (dP - rowsum(dP·P)) | Jacobian-vector product를 O(n)으로 |
| Scaling backward | dS / sqrt(d) | 상수 나누기 그대로 전달 |
| 검증 방법 | perturbation | 원소 하나를 ε 흔들어 추적 |

## FlashAttention과의 관계

위 backward 전체를 하나의 fused kernel에서 타일 단위로 처리하는 것이 FlashAttention이다.
중간 행렬 P, S를 메모리에 저장하지 않고, forward에서 `logsumexp`만 저장한 뒤
backward에서 P를 on-the-fly로 재계산하여 메모리를 O(n²)에서 O(n)으로 줄인다.
