# Preliminary Analysis: Residual Dominance in Transformer-based Sequential Recommendation

事前分析ページ用の結果まとめ。2つの独立した分析で「Transformer 系逐次推薦モデルにおける残差支配 (residual dominance) 問題」を定量化する。

- **混合率分析 (Kobayashi EMNLP 2020/2021)**: 残差接続+LN が Attention の文脈情報をどれだけ抑制するかの**構造的**証拠 → BSARec & SASRec
- **HRLI@1 (Oh & Cho RecSys 2024)**: モデルが最終アイテムを返すだけになる現象の**行動的**証拠 → BSARec, SASRec, KDStudent, GRU4Rec

スクリプト:
- [BSARec/src/analyze_mixing_ratio.py](BSARec/src/analyze_mixing_ratio.py)
- [BSARec/src/analyze_hrli.py](BSARec/src/analyze_hrli.py)

生データ:
- [BSARec/src/results/mixing_ratio.md](BSARec/src/results/mixing_ratio.md) / [.json](BSARec/src/results/mixing_ratio.json)
- [BSARec/src/results/hrli.md](BSARec/src/results/hrli.md) / [.json](BSARec/src/results/hrli.json)

図:
- [BSARec/src/results/mixing_ratio_last_summary.png](BSARec/src/results/mixing_ratio_last_summary.png) — スライド向け（layer 平均）
- [BSARec/src/results/mixing_ratio_last.png](BSARec/src/results/mixing_ratio_last.png) — 詳細（layer 別）
- [BSARec/src/results/mixing_ratio_all_summary.png](BSARec/src/results/mixing_ratio_all_summary.png), [BSARec/src/results/mixing_ratio_all.png](BSARec/src/results/mixing_ratio_all.png) — sequence average 版

---

## 分析1: 混合率分析 (BSARec & SASRec)

### スコープ
`_modules.py` の `MultiHeadAttention.forward` 内部のみを対象。
```
pre  = self.dense(context_layer)                          ← 残差前 (Attention 出力 W_O 通過後)
post = self.LayerNorm(self.out_dropout(pre) + input_tensor) ← 残差+LN 後
```
- BSARec: `model.item_encoder.blocks[l].layer.attention_layer` (BSARecLayer 内部の Self-Attention 経路)
- SASRec: `model.item_encoder.blocks[l].layer` (TransformerBlock 内部の MultiHeadAttention)

**含めないもの**: BSARecLayer の α-mixing (FFT 経路) / 各 block 外側 FFN。

### 混合率の定義
各位置 i での「自分以外の位置からの寄与の割合」。Kobayashi 流に、`pre` を per-source contribution `f_j = attn_probs[i,j] · value[j] @ W_O` に分解し L2 ノルム比で計算。`post` は LayerNorm の局所線形化 (γ/σ·(f - mean(f))) を介して decomposition を伝播。

### 結果（最終位置 = 推薦に使う位置）

| Model | Dataset | Layer | Pre (Attn-only) | Post (Attn+Res+LN) | **Suppression** |
|---|---|---|---:|---:|---:|
| BSARec | Beauty | 0 | 0.9336 | 0.2888 | **69.1%** |
| BSARec | Beauty | 1 | 0.9337 | 0.3721 | **60.2%** |
| BSARec | LastFM | 0 | 0.9781 | 0.6792 | **30.6%** |
| BSARec | LastFM | 1 | 0.9745 | 0.6590 | **32.4%** |
| BSARec | ML-1M  | 0 | 0.9597 | 0.6881 | **28.3%** |
| BSARec | ML-1M  | 1 | 0.9147 | 0.5029 | **45.0%** |
| SASRec | Beauty | 0 | 0.9247 | 0.2134 | **76.9%** |
| SASRec | LastFM | 0 | 0.9627 | 0.2328 | **75.8%** |
| SASRec | LastFM | 1 | 0.9696 | 0.2051 | **78.8%** |
| SASRec | ML-1M  | 0 | 0.9712 | 0.5689 | **41.4%** |

### Layer 平均（スライド用）

| Dataset | BSARec Pre | BSARec Post | **BSARec Supp.** | SASRec Pre | SASRec Post | **SASRec Supp.** |
|---|---:|---:|---:|---:|---:|---:|
| Beauty | 0.934 | 0.331 | **-65%** | 0.925 | 0.213 | **-77%** |
| LastFM | 0.976 | 0.669 | **-31%** | 0.966 | 0.219 | **-77%** |
| ML-1M  | 0.937 | 0.595 | **-36%** | 0.971 | 0.569 | **-41%** |

### 解釈
- **Pre は全ケースで 0.92-0.98**: Self-Attention 単体では他位置の情報がほぼ全て使われている（健全な Attention の動作）。
- **Post は 0.21-0.69 まで低下**: 残差 + LN が他位置寄与を大幅に削っている。
- **抑制率は 28-79%**: Kobayashi らの知見（35-65%）の幅と整合・やや上回るケースもあり。
- **SASRec > BSARec**: SASRec は全データセットで抑制率が高く、特に LastFM では 77% vs BSARec 31%。**BSARec の FFT 経路が部分的に文脈情報を温存**していると解釈できる。
- **データセット依存性**: ML-1M は両モデルとも抑制が弱い（28-45%）。Beauty/LastFM は強く抑制されている。

---

## 分析2: HRLI@1 (Hidden Representation Last-item Influence)

### 定義
モデルの Top-1 推薦が **入力シーケンスの最終アイテム** と一致する割合（filter-seen 適用なし、Oh & Cho 2024 準拠）。HRLI@1 が高い = モデルが直前アイテムをそのまま返している = 残差支配の間接証拠。

### 結果

| Model | Beauty | LastFM | ML-1M |
|---|---:|---:|---:|
| **SASRec** (Transformer)     | **0.9441** | **0.9413** | **0.9652** |
| **BSARec** (Transformer)     | **0.8453** | **0.6606** | **0.2884** |
| KDStudent (GRU + KD from BSARec) | 0.3085 | 0.0450 | 0.1055 |
| GRU4Rec  (Pure GRU)          | 0.0080 | 0.0101 | 0.0964 |

### 解釈
- **SASRec は 94-97% で最終アイテムをそのまま返している** — 混合率分析の高い抑制率（77% 平均）と整合。
- **BSARec は 29-85%** — 抑制率の幅（28-69%）と粗く対応。FFT 経路の補正で SASRec より緩和されている。
- **GRU 系は 1-10%** — 構造的に最終アイテム依存しない。
- **KDStudent < BSARec teacher** — KD で性能を取り込みつつ、最終アイテム依存は引き継がない。GRU の inductive bias が効いている。

---

## 2つの分析の整合性

| | Beauty | LastFM | ML-1M |
|---|---|---|---|
| BSARec 混合率の抑制率 (layer 平均, last) | 65% | 31% | 36% |
| SASRec 混合率の抑制率 (layer 平均, last) | 77% | 77% | 41% |
| BSARec HRLI@1                | 0.85  | 0.66  | 0.29  |
| SASRec HRLI@1                | 0.94  | 0.94  | 0.97  |
| α-scaling 既知の最適 α       | ~0.75 | 0     | ~1.5  |

**整合パターン**: SASRec の抑制率は全データセットで BSARec より高く、HRLI@1 も SASRec が常に高い。混合率と HRLI@1 はモデル間比較では順序が一致している。

**注意**: データセット内では一致しない。例えば BSARec の Beauty (抑制 65%) vs LastFM (抑制 31%) で、抑制率は Beauty が高いのに HRLI@1 も Beauty が高い (0.85 vs 0.66) — 単純な「抑制が強いほど HRLI が高い」ではなく、データ密度・系列長・モデル学習の相互作用が絡む。

---

## スライド作成のための提案

### 推奨構成: 2ページに分ける

**ページ 2-A: 混合率分析** — 「Attention の文脈情報が構造的に削られる」

主役の図: `mixing_ratio_last_summary.png`
- 一目で「Pre は高い、Post は低い、SASRec の方が抑制が強い」が伝わる
- データセット依存性も視覚的に伝わる

キーメッセージ:
- **「Attention は本来 0.92-0.97 の比率で他位置の情報を集めている」**
- **「しかし残差+LN を通った後は 0.21-0.69 まで圧縮 — 28-79% の文脈情報が抑制」**
- **「SASRec は全データセットで抑制が強く、BSARec は FFT 経路で部分緩和」**

**ページ 2-B: HRLI@1** — 「結果として Transformer は最終アイテムを返すだけになりがち」

主役の図: HRLI@1 の 4×3 棒グラフ（要作成）
- Transformer 系（SASRec/BSARec）を赤系、GRU 系（KDStudent/GRU4Rec）を青系で対比

キーメッセージ:
- **「SASRec は 94-97% が直前アイテムそのまま — 残差支配の極端な現れ」**
- **「GRU4Rec は 1-10% — 構造的に依存しない」**
- **「→ 3 ページ目で本研究の解決策（KD で GRU 生徒に文脈情報を直接転移）」**

### 1ページにまとめる場合

スライドを1枚にまとめるなら左に混合率の summary chart、右に HRLI@1 の表 + 一言メッセージ「混合率の構造的抑制が、HRLI@1 の極端な最終アイテム依存として行動レベルで現れる」。
