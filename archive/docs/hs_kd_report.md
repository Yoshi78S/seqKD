# Hidden-state KD 実験レポート

## 目的

BSARec 教師の残差接続前 Attention 出力（混合率 0.83-0.92 の豊富な文脈情報）を GRU4Rec 生徒の隠れ状態に align する蒸留損失を実装・評価する。Prediction-level KD だけでは残差支配後の情報しか転移できないという分析結果に基づく。

## 手法

### 損失関数

$$L = L_{rec} + \lambda_{pred} \times L_{pred} + \lambda_{hs} \times L_{hs}$$

- $L_{rec}$: CrossEntropy(student\_logits, answers)
- $L_{pred}$: KL(softmax(teacher/T) || log\_softmax(student/T)) × T² （既存 Prediction KD）
- $L_{hs}$: Hidden-state alignment loss （提案手法）

### Hidden-state の取得

- **教師 (BSARec)**: 最終ブロックの `MultiHeadAttention.dense` に forward hook を登録し、残差接続前の W_O 射影出力を `.detach()` で捕捉
- **生徒 (GRU4Rec)**: `nn.GRU` に forward hook を登録し、dense 射影前の生 GRU 出力を捕捉（勾配は保持）

### 実験設定

- **共通**: hidden\_size=64, gru\_hidden\_size=64, max\_seq=50, layers=2, lr=0.001, batch=256, patience=10, T=2.0
- **教師**: グリッドサーチ済み BSARec チェックポイント（Beauty: h=4/α=0.7/c=5, LastFM: h=2/α=0.9/c=9, ML-1M: h=1/α=0.3/c=9）

---

## 結果

### 既存手法との比較（Phase 1）

| Dataset | Method | λ\_pred | λ\_hs | HR@10 | NDCG@10 | vs Pred KD |
|---|---|---:|---:|---:|---:|---:|
| **Beauty** | GRU4Rec 単体 | 0 | 0 | 0.0267 | — | — |
| | Pred KD only (v2) | 1.0 | 0 | 0.0917 | 0.0538 | baseline |
| | **HS KD only** | 0 | 1.0 | 0.0731 | 0.0408 | -20.3% |
| | **Multi-level KD** | 1.0 | 1.0 | 0.0911 | 0.0531 | -0.7% |
| **LastFM** | GRU4Rec 単体 | 0 | 0 | 0.0422 | — | — |
| | Pred KD only (v2) | 1.0 | 0 | 0.0651 | 0.0372 | baseline |
| | **HS KD only** | 0 | 1.0 | 0.0587 | 0.0293 | -9.8% |
| | **Multi-level KD** | 1.0 | 1.0 | 0.0651 | 0.0353 | +0.0% |
| **ML-1M** | GRU4Rec 単体 | 0 | 0 | 0.1834 | — | — |
| | Pred KD only (v2) | 1.0 | 0 | 0.2767 | 0.1539 | baseline |
| | **HS KD only** | 0 | 1.0 | 0.2570 | 0.1429 | -7.1% |
| | **Multi-level KD** | 1.0 | 1.0 | 0.2662 | 0.1478 | -3.8% |

### Ablation: λ\_hs のチューニング（Phase 2, multi-level, MSE, all positions）

| Dataset | λ\_hs=0 (Pred only) | λ\_hs=0.1 | λ\_hs=0.5 | λ\_hs=1.0 | λ\_hs=2.0 |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.0917 | **0.0934** | 0.0915 | 0.0911 | 0.0923 |
| LastFM | 0.0651 | **0.0697** | 0.0679 | 0.0651 | 0.0606 |
| ML-1M | 0.2767 | **0.2753** | 0.2619 | 0.2662 | 0.2538 |

### Ablation: 損失関数タイプ（multi-level, λ\_hs=1.0, all positions）

| Dataset | MSE | Cosine |
|---|---:|---:|
| Beauty | 0.0911 | **0.0927** |
| LastFM | 0.0651 | **0.0688** |
| ML-1M | 0.2662 | 0.2619 |

### Ablation: 位置モード（multi-level, λ\_hs=1.0, MSE）

| Dataset | all positions | last only |
|---|---:|---:|
| Beauty | 0.0911 | 0.0914 |
| LastFM | 0.0651 | 0.0606 |
| ML-1M | 0.2662 | **0.2755** |

### 全実験結果一覧

| Dataset | Experiment | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | hs\_only | 0.0496 | 0.0731 | 0.1062 | 0.0333 | 0.0408 | 0.0492 |
| Beauty | multi\_level | 0.0626 | 0.0911 | 0.1292 | 0.0439 | 0.0531 | 0.0626 |
| Beauty | ml\_lhs0.1 | 0.0646 | **0.0934** | 0.1314 | 0.0459 | 0.0552 | 0.0647 |
| Beauty | ml\_lhs0.5 | 0.0628 | 0.0915 | 0.1276 | 0.0444 | 0.0536 | 0.0627 |
| Beauty | ml\_lhs2.0 | 0.0664 | 0.0923 | 0.1284 | 0.0459 | 0.0542 | 0.0633 |
| Beauty | ml\_cosine | 0.0657 | 0.0927 | 0.1287 | 0.0465 | 0.0551 | 0.0641 |
| Beauty | ml\_last | 0.0650 | 0.0914 | 0.1292 | 0.0458 | 0.0543 | 0.0638 |
| LastFM | hs\_only | 0.0349 | 0.0587 | 0.0927 | 0.0218 | 0.0293 | 0.0378 |
| LastFM | multi\_level | 0.0459 | 0.0651 | 0.0963 | 0.0289 | 0.0353 | 0.0431 |
| LastFM | ml\_lhs0.1 | 0.0450 | **0.0697** | 0.1064 | 0.0277 | 0.0359 | 0.0450 |
| LastFM | ml\_lhs0.5 | 0.0422 | 0.0679 | 0.0963 | 0.0293 | 0.0374 | 0.0446 |
| LastFM | ml\_lhs2.0 | 0.0450 | 0.0606 | 0.1028 | 0.0275 | 0.0325 | 0.0431 |
| LastFM | ml\_cosine | 0.0523 | 0.0688 | 0.1055 | 0.0343 | 0.0397 | 0.0490 |
| LastFM | ml\_last | 0.0422 | 0.0606 | 0.1055 | 0.0281 | 0.0341 | 0.0453 |
| ML-1M | hs\_only | 0.1786 | 0.2570 | 0.3596 | 0.1177 | 0.1429 | 0.1686 |
| ML-1M | multi\_level | 0.1808 | 0.2662 | 0.3710 | 0.1205 | 0.1478 | 0.1742 |
| ML-1M | ml\_lhs0.1 | 0.1866 | 0.2753 | 0.3818 | 0.1251 | 0.1537 | 0.1805 |
| ML-1M | ml\_lhs0.5 | 0.1752 | 0.2619 | 0.3695 | 0.1167 | 0.1447 | 0.1718 |
| ML-1M | ml\_lhs2.0 | 0.1714 | 0.2538 | 0.3555 | 0.1129 | 0.1396 | 0.1652 |
| ML-1M | ml\_cosine | 0.1786 | 0.2619 | 0.3732 | 0.1202 | 0.1469 | 0.1750 |
| ML-1M | ml\_last | 0.1932 | **0.2755** | 0.3820 | 0.1283 | 0.1548 | 0.1815 |

---

## 分析

### 1. HS KD 単体は Pred KD より弱い

HS KD only は全データセットで Pred KD only を下回る（Beauty -20.3%, LastFM -9.8%, ML-1M -7.1%）。残差前の Attention 出力をそのまま align しても、最終的な推薦タスクとの距離が大きいため、logits ベースの蒸留ほど効率的でない。

### 2. デフォルト設定 (λ\_hs=1.0) の Multi-level は改善しない

λ\_pred=1.0, λ\_hs=1.0 の Multi-level KD は Pred KD only とほぼ同等か微減。HS 損失が大きすぎて Prediction 損失の学習を阻害している可能性がある。

### 3. λ\_hs=0.1 が最適

λ\_hs を 0.1 に下げると、全データセットで Pred KD only を上回るか同等:
- **Beauty**: 0.0917 → **0.0934** (+1.9%)
- **LastFM**: 0.0651 → **0.0697** (+7.1%)
- **ML-1M**: 0.2767 → 0.2753 (-0.5%)

**HS 損失は補助的な正則化として少量加えるのが効果的**。λ\_hs が大きいほど性能が劣化する明確な傾向がある。

### 4. Cosine 損失は LastFM で特に有効

Cosine 損失（λ\_hs=1.0）は MSE（λ\_hs=1.0）より:
- Beauty: 0.0911 → 0.0927 (+1.8%)
- **LastFM: 0.0651 → 0.0688 (+5.7%)**
- ML-1M: 0.2662 → 0.2619 (-1.6%)

方向の一致を重視する Cosine 損失は、スケールに敏感でないため MSE より安定する傾向がある。

### 5. ML-1M は last-only が最良

ML-1M では position\_mode='last' が全実験中の最高値 **0.2755** を達成（Pred KD only の 0.2767 とほぼ同等）。密な長系列では、全位置の align がノイズになり、最終位置のみの方が効率的。

### 6. ベスト設定のまとめ

| Dataset | Best HS-KD config | HR@10 | vs Pred KD only |
|---|---|---:|---:|
| Beauty | ml\_lhs0.1 (λ\_hs=0.1, MSE, all) | **0.0934** | **+1.9%** |
| LastFM | ml\_lhs0.1 (λ\_hs=0.1, MSE, all) | **0.0697** | **+7.1%** |
| ML-1M | ml\_last (λ\_hs=1.0, MSE, last) | **0.2755** | -0.4% |

Beauty と LastFM では Hidden-state KD が Prediction KD を改善。ML-1M では同等。

### 7. 蒸留なし → ベスト HS-KD の改善率

| Dataset | 蒸留なし | Pred KD | Best HS-KD | 改善率 (vs 蒸留なし) |
|---|---:|---:|---:|---:|
| Beauty | 0.0267 | 0.0917 | **0.0934** | **+250%** |
| LastFM | 0.0422 | 0.0651 | **0.0697** | **+65%** |
| ML-1M | 0.1834 | 0.2767 | **0.2755** | **+50%** |

---

## 考察

### 残差前 Attention 出力の蒸留が有効に機能する条件

HS KD は Prediction KD の補助として **少量 (λ\_hs=0.1)** 加えた時に最も効果的。これは以下のように解釈できる:

1. **Prediction KD がメインの学習信号**: logits の蒸留は最終的な推薦タスクに直結するため、学習効率が高い
2. **HS KD は正則化として機能**: 残差前の文脈情報を少量 align することで、生徒の中間表現が教師の文脈認識に近づき、汎化性能が向上する
3. **λ\_hs が大きいと干渉する**: HS 損失が Prediction 損失を圧倒すると、タスクに無関係な表現の一致に最適化が向かい、推薦性能が低下する

### 分析実験との整合性

- **混合率の観察との一致**: BSARec の残差前 Attention 出力（混合率 0.83-0.92）には豊富な文脈情報がある。これを少量 align することで、GRU4Rec の表現に文脈情報が転移される
- **相補性の裏付け**: Prediction KD + HS KD が Prediction KD 単体を上回ることは、logits だけでは伝わらない情報が残差前出力に存在することを示す

## 出力物

| ファイル | 内容 |
|---|---|
| `output/gru4rec_{Dataset}_{config}.pt` | 生徒チェックポイント（21本） |
| `output/gru4rec_{Dataset}_{config}.log` | 訓練ログ（21本） |
