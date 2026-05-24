# KDStudent 実験レポート

## 目的

BSARec 教師からの蒸留に最適化された軽量生徒モデル（KDStudent）を設計・評価する。
GRU4Rec の構造的限界（pos_emb なし, LN なし, FFN なし, flat GRU）を解消し、
教師のブロック構造と 1:1 対応させることで hidden-state alignment の効率を高める。

## アーキテクチャ

```
KDStudentModel:
  item_emb + pos_emb → LayerNorm → Dropout   （BSARec と同じ前処理）

  StudentBlock × 2:
    GRU(1層, hidden=64, bias=False) → Dropout → LayerNorm   （Attention の代替、残差なし）
    FeedForward(64→256→64, GELU, 残差+LN)                    （BSARec の FFN と同一構造）

  Prediction: output[:, -1, :] @ item_emb.T → logits
```

### GRU4Rec との構造的差分

| | GRU4Rec | KDStudent |
|---|---|---|
| Position embedding | なし | あり |
| LayerNorm | なし | あり（GRU後 + FFN内） |
| FFN | なし（dense のみ） | あり（GELU + 2層Linear + 残差+LN） |
| ブロック構造 | flat GRU(2層) | ブロック×2（教師と1:1対応） |
| 推論計算量 | O(n) | O(n) |

## 実験設定

- **共通**: hidden_size=64, max_seq=50, num_layers=2, lr=0.001, batch=256, patience=10
- **Dropout**: Beauty/LastFM=0.5, ML-1M=0.2
- **KD**: T=2.0, λ_kd=1.0（Pred KD）, λ_hs=0.1（HS-KD, MSE, all positions）
- **教師**: グリッドサーチ済み BSARec チェックポイント

### パラメータ数

| Dataset | KDStudent | GRU4Rec | BSARec教師 |
|---|---:|---:|---:|
| Beauty | 893,696 | 831K | ~900K |
| LastFM | 352,576 | — | — |
| ML-1M | 337,856 | — | — |

（差分は item_embedding サイズの違い。エンコーダ構造は共通）

---

## 結果

### KDStudent 全結果

| Dataset | Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | Standalone | 0.0393 | 0.0653 | 0.0967 | 0.0252 | 0.0335 | 0.0414 |
| Beauty | Pred KD | 0.0439 | 0.0712 | 0.1102 | 0.0286 | 0.0373 | 0.0471 |
| Beauty | **Multi-level KD** | **0.0475** | **0.0732** | **0.1114** | **0.0316** | **0.0398** | **0.0494** |
| LastFM | Standalone | 0.0303 | 0.0505 | 0.0817 | 0.0213 | 0.0277 | 0.0355 |
| LastFM | **Pred KD** | **0.0450** | **0.0706** | **0.1009** | **0.0298** | **0.0380** | **0.0457** |
| LastFM | Multi-level KD | 0.0385 | 0.0596 | 0.0972 | 0.0271 | 0.0338 | 0.0433 |
| ML-1M | **Standalone** | **0.2018** | **0.2937** | **0.3995** | **0.1358** | **0.1655** | **0.1921** |
| ML-1M | Pred KD | 0.1977 | 0.2886 | 0.3980 | 0.1316 | 0.1609 | 0.1884 |
| ML-1M | Multi-level KD | 0.1942 | 0.2864 | 0.3965 | 0.1311 | 0.1607 | 0.1884 |

### GRU4Rec との比較（HR@10）

| Dataset | GRU4Rec 単体 | GRU4Rec+PredKD | GRU4Rec+ML-KD | KDStudent 単体 | KDStudent+PredKD | KDStudent+ML-KD | BSARec教師 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 0.0267 | 0.0917 | 0.0934 | 0.0653 | 0.0712 | **0.0732** | 0.0962 |
| LastFM | 0.0422 | 0.0651 | 0.0697 | 0.0505 | **0.0706** | 0.0596 | 0.0734 |
| ML-1M | 0.1834 | 0.2767 | 0.2755 | **0.2937** | 0.2886 | 0.2864 | 0.2755 |

### 単体性能の改善（KDStudent vs GRU4Rec、蒸留なし）

| Dataset | GRU4Rec | KDStudent | 改善率 |
|---|---:|---:|---:|
| Beauty | 0.0267 | 0.0653 | **+144.6%** |
| LastFM | 0.0422 | 0.0505 | **+19.7%** |
| ML-1M | 0.1834 | 0.2937 | **+60.1%** |

### 各手法の Best（HR@10）

| Dataset | GRU4Rec Best | KDStudent Best | BSARec教師 |
|---|---:|---:|---:|
| Beauty | 0.0934 (ML-KD) | 0.0732 (ML-KD) | 0.0962 |
| LastFM | 0.0697 (ML-KD) | 0.0706 (Pred KD) | 0.0734 |
| ML-1M | 0.2767 (Pred KD) | **0.2937 (Standalone)** | 0.2755 |

---

## 分析

### 1. KDStudent は GRU4Rec より大幅に高い単体性能

pos_emb + LN + FFN の追加により、蒸留なしでも全データセットで GRU4Rec を大きく上回る:
- Beauty: +144.6%（0.0267 → 0.0653）
- LastFM: +19.7%（0.0422 → 0.0505）
- ML-1M: +60.1%（0.1834 → 0.2937）

**ML-1M では BSARec 教師（0.2755）を単体で超えた**（0.2937, +6.6%）。

### 2. KD の効果はデータセットで大きく異なる

| Dataset | Standalone → Best KD | 改善率 | 有効な KD |
|---|---|---:|---|
| Beauty | 0.0653 → 0.0732 | +12.1% | Multi-level KD |
| LastFM | 0.0505 → 0.0706 | +39.8% | Pred KD |
| ML-1M | 0.2937 → 0.2937 | 0% | **KD は逆効果** |

- **Beauty**: Pred KD (+9.0%) も Multi-level KD (+12.1%) も有効
- **LastFM**: Pred KD が最も有効（+39.8%）、HS-KD を加えると逆に低下
- **ML-1M**: 既に教師を超えているため、KD が性能を劣化させる

### 3. GRU4Rec+KD vs KDStudent+KD: 逆転現象

| Dataset | GRU4Rec Best | KDStudent Best | 勝者 |
|---|---:|---:|---|
| Beauty | **0.0934** | 0.0732 | GRU4Rec |
| LastFM | 0.0697 | **0.0706** | KDStudent |
| ML-1M | 0.2767 | **0.2937** | KDStudent |

**Beauty では GRU4Rec+KD が KDStudent+KD を上回る**。これは予想外の結果。

解釈: GRU4Rec は単体性能が極めて低い（0.0267）ため、KD による改善余地が大きい（+250%）。一方 KDStudent は単体でそこそこ強い（0.0653）ため、KD の追加効果が小さい。GRU4Rec の「白紙」状態が逆に教師の知識を吸収しやすい可能性がある。

### 4. HS-KD（Multi-level）は KDStudent では限定的

| Dataset | Pred KD only | Multi-level KD | HS-KD の効果 |
|---|---:|---:|---:|
| Beauty | 0.0712 | **0.0732** | +2.8% |
| LastFM | **0.0706** | 0.0596 | **-15.6%** |
| ML-1M | 0.2886 | 0.2864 | -0.8% |

LastFM で HS-KD が大きくマイナス。KDStudent の GRU 出力と BSARec の Attention 出力の align が、Pred KD の学習信号を阻害している可能性がある。

### 5. ML-1M での教師超え

KDStudent 単体が BSARec 教師を超えた初めてのケース:
- KDStudent standalone: **0.2937**
- BSARec 教師: 0.2755（+6.6%）
- GRU4Rec + Pred KD: 0.2767

ML-1M は密な長系列データセットで、GRU の逐次処理が Attention の O(n²) より効率的に機能する。pos_emb + LN + FFN の追加で GRU の弱点を補いつつ、GRU の長系列処理の強みを活かせている。

---

## 考察

### KD の改善幅は生徒の単体性能に依存する

単体性能が低い生徒ほど KD による改善余地が大きい:

- **GRU4Rec（単体が弱い）**: Beauty で KD により +250% 改善
- **KDStudent（単体が強い）**: Beauty で KD により +12% 改善
- **KDStudent on ML-1M（教師超え）**: KD は逆効果

これはトレードオフではなく、**単体性能が低いほど教師から学べる余地が大きい**という自然な帰結。重要なのは最終性能であり、KDStudent は LastFM と ML-1M で GRU4Rec+KD の最終性能を上回っている。

Beauty で GRU4Rec+KD（0.0934）が KDStudent+KD（0.0732）を上回る原因は、KD の改善余地の違いだけでなく、KD の HP（λ_kd, T）が GRU4Rec 向けに調整されている点や、Beauty のデータ特性との相性も考えられる。HP チューニングで改善の余地がある。

### 実用的な指針

- **アーキテクチャ設計が最も重要**: pos_emb + LN + FFN の追加だけで ML-1M +60%、教師超えも達成
- **KD の効果は生徒の能力に依存**: 生徒が既に強い表現を学習できる場合、KD の追加効果は限定的
- **教師超えの条件**: 生徒のアーキテクチャが特定のデータ特性に合致する場合（ML-1M の長系列 + GRU）

---

## 出力物

| ファイル | 内容 |
|---|---|
| `output/kdstudent_{Dataset}_standalone.pt` | 単体学習チェックポイント（3本） |
| `output/kdstudent_{Dataset}_pred_kd.pt` | Pred KD チェックポイント（3本） |
| `output/kdstudent_{Dataset}_multilevel.pt` | Multi-level KD チェックポイント（3本） |
| `output/kdstudent_{Dataset}_{config}.log` | 訓練ログ（9本） |
