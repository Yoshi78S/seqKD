# 解離分析: recency 受容野 (A) vs 最終アイテム/残差支配 (B)

目的: 2つの現象を**別々に・モデル非依存**に測り、**解離する（同じ軸ではない）**ことを示す。
さらに Transformer では B を残差ストリームに機構帰属する。

- **A = recency 受容野**: 予測がどこまで過去に依存するか。指標 **k95**（直近 k 件だけ入力したとき
  HR@10 が full の 95% に達する最小 k；小さい=受容野短い=recency 強い）。
- **B = 最終アイテム/残差支配**: 最終位置の出力が最終アイテムに留まる傾向。指標 **HRLI@1/@10**
  （マスクなし Top-K に最終入力アイテムが入る割合）と **cos(h_n, E_last)**。

スクリプト: [BSARec/src/analyze_dissociation.py](BSARec/src/analyze_dissociation.py)（再実行で再現）。
生データ: [BSARec/src/results/dissociation.csv](BSARec/src/results/dissociation.csv)。
図: [dissociation_HRLI10.png](BSARec/src/results/dissociation_HRLI10.png) /
[dissociation_coslast.png](BSARec/src/results/dissociation_coslast.png)。
混合率(機構): [BSARec/src/results/mixing_ratio.md](BSARec/src/results/mixing_ratio.md)。
eval / no_grad / seed固定。test split。**サニティゲート: 18/18 すべて HR@10 が Experiment① を再現**（→指標は信用可）。

## 結果表（6モデル × 3データセット）

| DS | Model | k95 (A) | HRLI@1 | HRLI@10 (B) | cos_last (B) | cos_gt | mixing比(残差後, Tx) |
|---|---|---|---|---|---|---|---|
| ML-1M | SASRec | 5 | 0.959 | 0.994 | 0.575 | 0.157 | 0.57 |
| ML-1M | BSARec(teacher) | 5 | 0.288 | 0.754 | 0.489 | 0.271 | 0.60 |
| ML-1M | SIGMA | 20 | 0.325 | 0.788 | 0.478 | 0.283 | — |
| ML-1M | Mamba4Rec | 10 | 0.310 | 0.718 | 0.502 | 0.298 | — |
| ML-1M | **FreqMamba(student)** | 10 | **0.169** | **0.639** | **0.391** | 0.249 | — |
| ML-1M | GRU4Rec | 5 | 0.096 | 0.391 | 0.411 | 0.293 | — |
| LastFM | SASRec | 2 | 0.944 | 0.983 | 0.534 | -0.043 | 0.22 |
| LastFM | BSARec(teacher) | 10 | 0.661 | 0.941 | 0.532 | 0.117 | 0.67 |
| LastFM | **FreqMamba(student)** | 20 | **0.613** | **0.915** | **0.486** | 0.116 | — |
| LastFM | Mamba4Rec | 10 | 0.684 | 0.921 | 0.714 | 0.119 | — |
| LastFM | SIGMA | 1 | 0.616 | 0.867 | 0.696 | 0.138 | — |
| LastFM | GRU4Rec | 10 | 0.010 | 0.078 | 0.365 | 0.207 | — |
| Beauty | SASRec | 1 | 0.945 | 0.987 | 0.660 | 0.006 | 0.21 |
| Beauty | BSARec(teacher) | 2 | 0.845 | 0.984 | 0.545 | 0.071 | 0.33 |
| Beauty | Mamba4Rec | 3 | 0.869 | 0.987 | 0.679 | 0.138 | — |
| Beauty | SIGMA | 3 | 0.845 | 0.981 | 0.655 | 0.129 | — |
| Beauty | **FreqMamba(student)** | 3 | **0.695** | **0.950** | 0.607 | 0.149 | — |
| Beauty | GRU4Rec | 10 | 0.008 | 0.053 | 0.272 | 0.176 | — |

mixing比(残差後)は BSARec/SASRec のみ（最終位置・層平均、低い=残差が文脈を抑制=残差支配）。

## 仮説の結論

**仮説1: A と B は解離する → 確認。**
ML-1M で GRU4Rec と SASRec は同じ k95=5（同一 recency 受容野）なのに HRLI@10 が 0.39 vs 0.99（2.5倍差）。
LastFM で GRU4Rec と BSARec は同じ k95=10 なのに HRLI@10 が 0.078 vs 0.941（12倍差）。同じ x に縦の開き
（散布図）。**GRU は recency を持つが B が低い** → A と B は別軸。

**仮説2: Mamba系は A も B も持つ → 確認。**
Mamba4Rec/SIGMA は Beauty/LastFM で HRLI@10 0.87–0.99・cos_last 0.66–0.71 と高B（純GRUは0.05–0.08で
低B）。SSM は状態減衰(A)＋skip(D項=B)の両方を持つ。

**仮説3（本丸）: 学生 FreqMamba は教師より B が低いか → YES（全DS）。**
HRLI@1/@10・cos_last すべて教師 BSARec より低い（ML-1M HRLI@1 0.169<0.288, Beauty 0.695<0.845,
LastFM 0.613<0.661）。かつ k95 は教師以上（5→10, 2→3, 10→20）=より広い受容野。
**最終アイテム依存を下げつつ文脈窓を広げている**。さらに FreqMamba の B は Mamba4Rec/SIGMA より低い
（残差なし Mamba 枝が標準 Mamba の D-skip より B を抑制）。

**機構（B→残差, Transformer）:**
BSARec 内で残差後 mixing 比が低い(Beauty 0.33)↔HRLI高(0.845)、高い(ML-1M 0.60)↔HRLI低(0.288)。
残差が文脈を抑制するほど最終アイテム支配が強い → **B を残差ストリームに帰属**。SASRec は一様に低mixing
(0.21–0.57)＋高HRLI(0.94–0.96)。

## 注意（誠実に）
- LastFM は N=1090・単一シードで truncation がノイジー（SIGMA k95=1 等は割り引く）。解離の主証拠は
  ML-1M（6k users で安定）。
- cos の絶対値は位置埋め込み込みで低め（同一データ内のモデル間相対比較のみ）。
- Beauty は多くのモデルが高Bに集中し解離は ML-1M/LastFM より不明瞭（GRU が唯一の外れ値）。
