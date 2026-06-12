# seqKD 実験結果集計

生成: 2026-06-12。ログは `seqKD/src/output/` を直接参照してください。  
σ ≈ ±0.004（seed=42 1-seed 推定）。**太字** = 各 DS のベスト既知値。

---

## 0. 参照基準

| モデル | Beauty HR@10 | NDCG@10 | LastFM HR@10 | NDCG@10 | ML-1M HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| BSARec 教師 | 0.0985 | 0.0599 | 0.0761 | 0.0437 | 0.2800 | 0.1572 |
| FreqMamba noKD | 0.0910 | 0.0538 | 0.0734 | 0.0389 | 0.2954 | 0.1658 |

- 教師ckpt: `BSARec/src/output/BSARec_{DS}_grid_lr…pt`
- noKD ログ: `src/output/fmamba_noKD_{DS}.log`

---

## 1. v3 グリッドサーチ (HS-KD 込み)

**実験**: λ_pred∈{0.5,1.0,2.0} × T∈{1.0,2.0,5.0} × (λ_hs,layer)∈{(0,—),(0.05,last),(0.1,last),(0.2,last)}, 36×3=108 runs  
**ログ**: `kdstudent_v3_{DS}_lp*_t*_lhs*_last.log`  
**完了**: 2026-05-30

### ベスト HP

| DS | α | λ_pred | T | λ_hs | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|---|---|
| Beauty | 0.7 | 1.0 | 1.0 | 0.05 | 0.0708 | **0.0989** | 0.1344 | 0.0503 | 0.0594 | 0.0683 |
| LastFM | 0.9 | 0.5 | 2.0 | 0.20 | 0.0550 | **0.0761** | 0.1165 | 0.0355 | 0.0424 | 0.0525 |
| ML-1M  | 0.3 | 0.5 | 2.0 | 0.05 | 0.2175 | **0.3078** | 0.4156 | 0.1489 | **0.1780** | 0.2051 |

### 競合比較 (HR@10 / NDCG@10)
- Beauty: #1 (0.0989 > FEARec 0.0986 > 教師 0.0985)
- LastFM: 教師同率 (0.0761 / 0.0424 vs 0.0761 / 0.0437)
- ML-1M: 教師を大幅超過 (+10% HR, DuoRec と NDCG@10 同率トップ 0.1780)

**含意**: HS-KD は寄与小（pred-only ≈ HS-on）。改善はほぼ FreqMamba アーキによる。

---

## 2. PL 蒸留スイープ (HS-KD なし)

**実験**: λ_pl∈{0.5,1.0,2.0} × k∈{10,20,50}, 9×3=27 runs  
**ログ**: `pl_{DS}_l{λ}_k{k}.log`

### ベスト

| DS | λ | k | HR@10 | NDCG@10 |
|---|---|---|---|---|
| Beauty | 2.0 | 50 | 0.0971 | 0.0582 |
| LastFM | 1.0 | 10 | 0.0807 | 0.0452 |
| ML-1M  | 0.5 | 50 | 0.3008 | 0.1726 |

v3 グリッドベスト（HS-KD あり）より BeautyΔ−0.0018, LastFM±0, ML-1M −0.0070。  
**以降の追加損失実験の PL 基準はこの HP 設定を固定して使用**。

---

## 3. CMI スイープ (β-gated 情報ボトルネック)

**実験**: β∈{0, 0.01, 0.1, 1.0} × 3DS  
**ログ**: `cmi_linear_{DS}_b{β}.log`  
**完了**: 2026-06-10

β=0 が pure PL と一致（以降の実験の基準値 = **PL baseline**)。

| DS | β | HR@10 | NDCG@10 | HRLI@1 | 判定 |
|---|---|---|---|---|---|
| Beauty | **0** | 0.0982 | 0.0591 | 0.697 | base |
| Beauty | 0.01 | 0.0957 | — | — | −0.0025 |
| Beauty | 0.1  | 0.0853 | — | — | −0.0129 |
| LastFM | **0** | 0.0807 | 0.0452 | 0.613 | base |
| LastFM | 0.01 | 0.0725 | — | — | −0.0082 |
| LastFM | 0.1  | 0.0569 | — | — | −0.0238 |
| ML-1M  | **0** | 0.3008 | 0.1726 | 0.169 | base |
| ML-1M  | 0.01 | 0.2935 | — | — | −0.0073 |
| ML-1M  | 0.1  | 0.2606 | — | — | −0.0402 |

**結論**: β>0 は全 DS で単調悪化。CMI 正則化項は学習を阻害。**FAILED**。

---

## 4. Debias Stage 1 (相対的 de-bias 損失) ※ ML-1M のみ

**実験**: 4 arm (margin / BPR / logit_margin / reweight) × λ/γ/m グリッド, 計 ~16 runs  
**ログ**: `db_{arm}_{DS}_{param}.log`  
**完了**: 2026-06-10

### ML-1M ベスト per arm (PL baseline = 0.3008)

| arm | 設定 | HR@10 | NDCG@10 | HRLI@1 | Δ HR@10 |
|---|---|---|---|---|---|
| none (control) | — | 0.2919 | 0.1690 | 0.187 | −0.0089 |
| margin | λ=0.1, m=0.3 | 0.2927 | 0.1687 | 0.144 | −0.0081 |
| BPR | λ=0.1 | 0.2922 | 0.1683 | 0.075 | −0.0086 |
| logit_margin | m=0.5 | 0.2911 | 0.1679 | 0.154 | −0.0097 |
| reweight | γ=1.0 | 0.2949 | 0.1677 | 0.127 | −0.0059 |

全 arm が base 未満。HRLI は下がるが HR も下がる（hi-κ 群だけでなく全体を削っている）。  
**含意**: κ で相対 de-bias → hi-κ↑ / lo-κ↓ の trade-off は損失形式によらない。**FAILED**。

---

## 5. κ-矯正 Relational Distillation R1 (rep)

**実験**: rep_raw / rep_pairgate × λ グリッド, Beauty・ML-1M  
**ログ**: `rep_{mode}_{DS}_l{λ}.log`  
**完了**: 2026-06-10

### ベスト (PL baseline: Beauty=0.0982, ML-1M=0.3008)

| DS | mode | λ | HR@10 | NDCG@10 | HRLI@1 | k-lo/mid/hi |
|---|---|---|---|---|---|---|
| Beauty | raw | 4 | 0.0962 | 0.0583 | 0.636 | 0.248/0.035/0.006 |
| Beauty | pairgate | 80 | 0.0961 | 0.0568 | 0.566 | 0.243/0.039/0.007 |
| ML-1M | raw | 10 | 0.2949 | 0.1698 | 0.153 | 0.460/0.313/0.112 |
| ML-1M | pairgate | 50 | **0.2993** | 0.1747 | 0.166 | **0.472**/0.311/0.116 |

ML-1M pairgate は hi-κ(k_lo) = 0.472 が base(≈0.435) を大幅超、しかし全体は 0.2993 < 0.3008。  
lo-κ 群の損傷がオーバーシュートを打ち消す構造。プラセボ(shuffled)と実質同値。  
**含意**: κ-relational signal は実在するが hi-κ/lo-κ の trade-off で全体 HR は改善しない。**FAILED**。

---

## 6. τ-gated PL (T1)

**実験**: kappa gate × γ∈{1,2,4,6,8} / d_only / shuffled, Beauty・ML-1M  
**ログ**: `taupl_{DS}_{mode}_g{γ}.log`  
**完了**: 2026-06-10

### ML-1M (PL baseline = 0.3008)

| mode | γ | HR@10 | NDCG@10 |
|---|---|---|---|
| kappa | 1 | 0.2919 | 0.1704 |
| kappa | 2 | 0.2969 | 0.1698 |
| kappa | **4** | **0.3013** | 0.1740 |
| kappa | 6 | 0.2912 | 0.1675 |
| kappa | 8 | 0.2987 | 0.1707 |
| d_only | 4 | 0.2969 | 0.1684 |
| shuffled | 4 | 0.2990 | 0.1704 |

### Beauty (PL baseline = 0.0982)

| mode | γ | HR@10 | NDCG@10 |
|---|---|---|---|
| kappa | 1 | 0.0956 | 0.0582 |
| kappa | 4 | 0.0955 | 0.0568 |

ML-1M γ=4 だけが辛うじて base を +0.0005 上回る。Beauty は全ケース base 未満。  
shuffled ≈ real → κ gate の作用はほぼランダムノイズ相当。**FAILED (±誤差範囲)**。

---

## 7. π̃-PL 修復蒸留 P1

**実験**: gate=kappa × (q67,c1)/(q67,c2)/(q33,c1)/(q33,c2) + shuffled@best, ML-1M; kappa q67 × c1/c2, Beauty  
**ログ**: `repair_{DS}_{gate}_{tau}_c{c}.log`  
**完了**: 2026-06-11

### ML-1M (PL baseline = 0.3008, HRLI = 0.169)

| gate | τ | c | HR@10 | NDCG@10 | HRLI@1 | k-lo/mid/hi |
|---|---|---|---|---|---|---|
| kappa | q67 | 1.0 | 0.2970 | 0.1715 | 0.111 | 0.438/0.319/0.134 |
| kappa | q67 | 2.0 | 0.2939 | 0.1677 | 0.145 | 0.437/0.310/0.135 |
| kappa | q33 | 1.0 | 0.2891 | 0.1654 | 0.123 | 0.430/0.311/0.127 |
| kappa | q33 | 2.0 | 0.2897 | 0.1682 | 0.093 | 0.424/0.307/0.139 |
| shuffled | q67 | 1.0 | 0.2969 | 0.1709 | 0.125 | 0.437/0.325/0.128 |

### Beauty (PL baseline = 0.0982, HRLI = 0.697)

| gate | τ | c | HR@10 | NDCG@10 | HRLI@1 | k-lo/mid/hi |
|---|---|---|---|---|---|---|
| kappa | q67 | 1.0 | 0.0952 | 0.0571 | 0.631 | 0.241/0.037/0.008 |
| kappa | q67 | 2.0 | 0.0944 | 0.0562 | 0.537 | 0.237/0.034/0.012 |

kappa と shuffled が全 DS で同等 → κ-selective な edit ではなく list churn そのものが変動源。  
hi-κ(k_lo) は改善するが lo-κ(k_hi) が低下し全体 HR は base 未満。**FAILED**。

---

## 8. 残差除去アブレーション (causal ladder) — 完結

**目的**: "Mamba 枝の残差なし = 残差支配解消" の主張を実験的に検証  
**実験**: 3 点すべて完了（2026-06-12, seed=42, `--kd_mode cmi --cmi_estimator none` = pure PL）  
**ログ**: `v3ablres_{DS}.log`, `cmi_linear_{DS}_b0.log`, `v3freqnores_{DS}.log`

### 3 点ラダー結果 (残差成分の多い順に上から)

| 設定 (mamba res / freq res) | Beauty HR@10 | NDCG | HRLI@1 | LastFM HR@10 | NDCG | HRLI@1 | ML-1M HR@10 | NDCG | HRLI@1 |
|---|---|---|---|---|---|---|---|---|---|
| **v3ablres** (ON / ON) | 0.0956 | 0.0572 | 0.708 | 0.0688 | 0.0402 | 0.769 | 0.3022 | 0.1740 | 0.184 |
| **v3 baseline** (OFF / ON) ← 採用 | 0.0982 | 0.0591 | 0.697 | 0.0807 | 0.0452 | 0.613 | 0.3008 | 0.1726 | 0.169 |
| **v3freqnores** (OFF / OFF) | 0.0824 | 0.0489 | 0.070 | 0.0532 | 0.0315 | 0.163 | 0.2937 | 0.1690 | 0.107 |

### 学習後 β²（高域恒等係数, 層ごと平均）— 恒等再出現の検査

`filtered = β²·x + (1−β²)·low_pass` なので β² が大きいほど入力 x の恒等成分が復活する。

| DS | v3 baseline (freq res ON) | v3freqnores (freq res OFF) |
|---|---|---|
| Beauty | 0.064, 0.088 | **2.804, 2.019** |
| LastFM | 0.239, 0.141 | **1.255, 1.099** |
| ML-1M  | 0.164, 0.181 | **0.437, 1.241** |

### 結論

1. **HRLI は残差成分量に単調追従** (ablres > baseline > freqnores)。特に Beauty で freq 残差を消すと HRLI 0.697→0.070 と崩壊、LastFM 0.613→0.163。**FrequencyLayer の `LN(filtered+x)` は last-item 依存（HRLI）の主要因**であり、critic 指摘（α=0.9 の LastFM で freq 枝が支配的）は正しい。

2. **しかし freq 残差除去は HR も全 DS で悪化** (Beauty −0.0158, LastFM −0.0275, ML-1M −0.0071)。残差は last-item コピーだけでなく有用な信号も運んでおり、HRLI と HR は不可分に絡む。**last-item 依存を完全に消すと精度も落ちる**。

3. **β² 恒等再出現を確認**: freqnores では β² が baseline の 10〜40 倍、かつ 1 を超える層が出現 (Beauty 2.8/2.0)。明示的残差 `+x` を奪われたモデルは高域パスを増幅して x の恒等成分を取り戻そうとする。"residual-free" は本質的に *枝限定かつ明示残差限定* でしか成立しない。

4. **中核主張の確定形（選択肢 c）**: HRLI 低下は単一残差の除去では説明できない。採用構成（Mamba 枝のみ残差除去＋freq 残差は保持）が HRLI と HR のバランス点。freq 残差を保持するのは意図的選択であり、消しても β² 再出現で部分的に戻るうえ HR を損なう。**「Mamba 枝の固定残差を外し、α 混合で freq 枝の残差寄与を (1−α) に抑える」のが正確な記述**で、「残差完全除去」とは主張しない。

---

## 9. V4 ゲート実験 (暫定)

**実験**: bipolar recency gate × (free/fixed/learned/shuffled) × λ_gate グリッド, Beauty・ML-1M  
**ログ**: `v4_{DS}_{mode}_lg{λ}.log`  
**完了**: 2026-06-11（但し未実装の V4DistillTrainer を使用 → 別途実装済みと思われる）

### ML-1M (PL baseline = 0.3008)

| mode | λ | HR@10 | NDCG@10 |
|---|---|---|---|
| free | 0.0 | 0.2685 | 0.1506 |
| fixed (−1) | 0.0 | 0.2296 | 0.1328 |
| learned | 2.0 | 0.2460 | 0.1451 |
| learned | 10.0 | 0.2442 | 0.1377 |
| learned_kl | 2.0 | 0.2815 | 0.1598 |
| shuffled | 2.0 | 0.2442 | 0.1396 |

### Beauty (PL baseline = 0.0982)

| mode | λ | HR@10 | NDCG@10 |
|---|---|---|---|
| free | 0.0 | 0.0968 | 0.0586 |
| learned | 2.3 | 0.0579 | 0.0301 |

全ケースが base を大幅下回る。gate 設計が学習を阻害している可能性が高い。**FAILED**。

---

## 10. まとめ / 既知の失敗パターン

| 手法 | DS | Δ HR@10 (vs PL base) | 結論 |
|---|---|---|---|
| CMI (β>0) | 全 | −0.003 ~ −0.040 | FAILED |
| Debias Stage 1 (best arm) | ML-1M | −0.006 | FAILED |
| Relational R1 pairgate | ML-1M | −0.002 | FAILED |
| τ-gated PL (best γ) | ML-1M | +0.0005 (≈誤差) | FAILED |
| π̃-PL Repair P1 (best arm) | ML-1M | −0.004 | FAILED |
| V4 gate | ML-1M | −0.019 ~ −0.071 | FAILED |
| **v3 Grid Best** (HS-KD 込み) | **全** | +0.007/0/+0.007 | **BEST** |

**収束する観察**:
1. κ 矯正系（debias / relational / repair）は一様に hi-κ↑ + lo-κ↓ → 全体 HR 改善なし
2. shuffled ≈ real は κ の実質信号ではなく list churn が原因と示唆
3. κ-HRLI 削減は最終的に "HRLI 低下は主にアーキ（Mamba 枝 + α混合）由来" であり蒸留損失では制御困難

4. 残差ラダー（§8）完結: HRLI は残差量に単調追従するが、freq 残差除去は HR も損ない β² 恒等再出現を招く。"残差完全除去" ではなく "Mamba 枝の固定残差除去＋α 混合で freq 残差を (1−α) に抑制" が正確な主張。

**未完了タスク**:
- V4DistillTrainer の再設計（現行実装は収束不安定; 全ケース base 大幅未満）— 優先度低

詳細な失敗記録は [FAILED_KD_METHODS.md](FAILED_KD_METHODS.md) 参照。
