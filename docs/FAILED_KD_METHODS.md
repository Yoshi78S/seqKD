# 失敗・不採用になった蒸留方法のまとめ

提案手法の探索過程で試して**効かなかった蒸留方法**の記録（負の結果＝論文の "what didn't work" / ablation 素材）。各項目: アイデア / 原因 / 判明方法。教師=BSARec, 生徒=FreqMamba v3（一部は旧 GRU 生徒 v1）。

最終的に残った（効いた）もの: **KL 予測蒸留**（Beauty/ML-1M ベスト）と **PL listwise ランキング蒸留**（LastFM で KL・教師を超え頑健）。RD-naive は不採用。

---

## 1. HS-KD（隠れ状態蒸留） — 不活性（寄与ほぼゼロ）
- **アイデア**: 教師の残差前 Attention 出力（`attention_layer.dense`）↔ 生徒の系列ミキサー出力（GRU/Mamba）を MSE でアライン。残差支配で抑制される文脈を直接転移する狙い。
- **結果**: 寄与がほぼゼロ。v1 ablation で D3(HS単独)≈D1(standalone)、D4(Pred+HS)≈D2(Pred単独)。`flat_gru`(HS無効)が ref を超えるDSも。v3 でも Pred-only ≈ Pred+HS。
- **判明**: grid/ablation 実走。**主因はアーキ、HSは付加価値なし**。
- 例外: LastFM のみ HS が僅かに効く（v3 best が λ_hs=0.2）。

## 2. CDD（Context-Direction Decorrelation） — 有害/無効
- **アイデア**: δ = h_pre − h_post（残差で失われた文脈方向）へ生徒表現を cos² で整列 + uniformity 損失。
- **結果**: Beauty −3.4% / LastFM −10.3% / ML-1M +1.8%。単独でも HS 併用でも改善せず。
- **判明**: 実走。δ 方向が学習信号として機能せず。→ 既定 λ_cdd=0 で封印（dead code）。

## 3. 適応ランキング蒸留 v1（Attention 温度ターゲット） — 診断で事前撤退
- **アイデア**: ρ=残差前 Attention エントロピーで順序依存度を測り、z_ord（Attention鋭化）/ z_set（Attention平坦化）を温度操作で作り、ρで補間して PL 蒸留。
- **原因**: **z_ord ≈ z_set**（Top-10 重複 78–92%）。Attention 温度は最終ロジットの Top-K をほとんど動かさない。さらに **ρ が狭い単峰（mean~0.1、二峰性なし）** で per-sample 適応の余地がない。残差由来の最終アイテム依存は「残差前 Attention」には写らない。
- **判明**: **事前診断（教師のみ）で grid 前に撤退**。コスト節約。

## 4. 適応ランキング蒸留 v2（残差前後 h_pre/h_post 補間） — 診断で事前撤退
- **アイデア**: z_set を残差前表現 h_pre 由来（route1=最終ブロック残差スキップ / route2=h_pre 直接内積）にして z_ord と補間。混合率 93% vs 33–67% で「構造的に必ず差が出る」想定。
- **原因**: **乖離と有効性が両立しない**。route1 は高α(Beauty/LastFM)で乖離せず（Top-10重複 0.80–0.93）、route2 は乖離するが **z_set が壊滅（HR −59〜−71%）**。どの経路も診断 A'(乖離)+B'(z_set有効性) を同時に通らない。
- **副次結論**: **残差前単体は良い予測を生まない＝残差は予測に必要**。混合率の sub-layer gap は最終ロジットに伝播しない（FFN+readout が吸収）。
- **判明**: 事前診断（A'/B'/C'）で grid 前に撤退。

## 5. 補完項蒸留（complementary, ρゲート） — 実走後に失敗
- **アイデア**: 主項=z_ord の PL ランキング + β·ρゲート·補完項（z_set の Top-K ∖ z_ord の Top-K を引き上げ）。診断で「set_only=補完余地」が低ρに局在（union-gain ML-1M +10.5%）を確認してから実装。
- **結果(ML-1M 梯子, rank_k=10)**: **補完項は害**。adaptive 0.2919 < PL-only(gate1) 0.2977、一律補完 gate0 0.2859 は **no-KD(0.2954) 以下**。ρゲートは一律よりマシ(5 vs 4 +0.006)だが補完自体が負。
- **原因**: 診断の union-gain（上限）は **学習で realize しない**。β を下げても gate1 に近づくだけで net 正にならない。
- **判明**: 実走（梯子 ablation: noKD/gate1/gate0/adaptive）。最良ケース ML-1M で否定 → Beauty/LastFM は診断で余地ゼロのため打ち切り。

## 6. RD-naive（pointwise ランキング蒸留） — 実走後に不採用
- **アイデア**: 教師 Top-K を重み付き正例として生徒スコアを引き上げ（`w=softmax(-rank/β)`、β大=一様）。RD (Tang&Wang 2018) 式。
- **結果(3DS, tuning済)**: **全DSで最下位KD**。ML-1M 0.2937 / LastFM 0.0706 は **no-KD 以下**。Beauty 0.0939 は no-KD は超えるが KL/PL 未満。
- **判明**: フェアな損失 ablation（KL vs RD vs PL, 各 best-of-grid）。pointwise は listwise(PL)/KL に劣る。

---

## 失敗から得た一般則
- **HS系・残差前表現の活用（HS-KD / CDD / 適応 v1・v2 / 補完）はことごとく効かない**。残差前の文脈情報は、最終ロジット/順位には伝播しない（FFN+readoutが吸収）、かつ残差は予測に必要。→ **「残差前を蒸留ターゲットにする」路線は全面撤退**。
- **効くのは予測（ロジット/順位）レベルの蒸留**: KL（分布全体, ピーク性能）と PL（順位のみ, 教師ミスキャリブレーションに頑健→LastFMでKL/教師超え）。
- **診断（教師のみ・安価）で事前に潰せた**: v1/v2 は grid を回さず撤退できた。補完・RD は実走が必要だった（診断の上限が realize するかは学習依存のため）。
