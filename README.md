# seqKD — Cross-architecture Knowledge Distillation for Sequential Recommendation

Transformer 教師（**BSARec**）の知識を、O(n) の軽量生徒 **FreqMamba** に蒸留する研究。
Transformer 系逐次推薦の構造的課題である **残差支配（最終アイテム依存）** を、
生徒アーキ（残差なし Mamba）と予測レベル蒸留で緩和しつつ、教師同等以上の精度を狙う。

- 教師: **BSARec**（AAAI 2024）。`h = α·FrequencyLayer(x) + (1-α)·Attention(x)` の2ブランチ融合。
- 生徒: **FreqMamba (KDStudent v3)**。BSARec の2ブランチ構造を継承し、O(n²) の Attention 枝のみを
  **残差なし Mamba** 枝に置換。
- 蒸留: **Plackett-Luce listwise ランキング蒸留**（採用）。順位のみ転写するため生徒次元に不変で、
  教師のミスキャリブレーションに頑健（後述の損失 ablation 参照）。

---

## 生徒モデル: FreqMamba (KDStudent v3)

各ブロック（×2, hidden=64）:
```
dsp = FrequencyLayer(x)              # BSARec と同一機構（low/high 分解 + 学習βで高域再調整, cutoff c）
ssp = LayerNorm(Mamba(x))           # 注意 → 残差なし Mamba（expand=1, d_state=16, d_conv=4）★固定残差なし
h   = α · dsp + (1-α) · ssp          # BSARec と同一の融合
x'  = LayerNorm(FFN(h) + h)          # 圧縮 FFN（64→128→64）
```
読み出し: `x'[:, -1, :] @ E_item^T`。実装: `src/model/kd_student_v3.py`。
本体パラメータ ~67K（item 埋め込み除く）で教師・旧 GRU 生徒より軽量、推論 O(n)。

## 蒸留損失

```
L = L_rec + λ · L_pred
```
- `L_rec`: 最終位置の全語彙 CrossEntropy（生徒の自律学習の錨）。
- `L_pred`: **PL listwise**（採用）= 教師 z_ord の Top-K 順位を生徒に listwise 転写。
  実装は `--kd_mode adaptive_rank_comp --gate_fixed 1.0`（補完項オフ＝純 PL）。
- 代替: **KL**（`--kd_mode kl`）。密データ（Beauty/ML-1M）ではピーク性能だが、教師が信頼しにくい
  LastFM では分布の歪みまで写して劣化。PL は順位のみ転写で頑健。

予測損失の公平 ablation（各損失 best-of-grid, HR@10）:

| Loss | Beauty | LastFM | ML-1M |
|---|---|---|---|
| KL | **0.0987** | 0.0725 | **0.3070** |
| **PL (採用)** | 0.0971 | **0.0807** | 0.3008 |
| RD-naive (pointwise) | 0.0939 | 0.0706 | 0.2937 |

→ KL がピーク（2/3）だが LastFM で no-KD すら下回る。**PL は全DSで安定し、LastFM で KL・教師を明確に超える**。
RD-naive は不採用。詳細は `FAILED_KD_METHODS.md`。

## 性能（教師 BSARec との比較, HR@10 / NDCG@10）

| Dataset | Teacher BSARec | FreqMamba + PL | FreqMamba 最良KD |
|---|---|---|---|
| Beauty | 0.0985 / 0.0599 | 0.0971 / 0.0582 | 0.0989 / 0.0594 (KL) |
| LastFM | 0.0761 / 0.0437 | **0.0807 / 0.0452** | 0.0807 / 0.0452 (PL) |
| ML-1M  | 0.2800 / 0.1572 | **0.3008 / 0.1726** | 0.3078 / 0.1780 (KL) |

- PL は **LastFM・ML-1M で教師超え**、Beauty は教師と僅差（~-1.4%）。最良KD（DSごと）では **3DS 全てで教師超え**。
- 13 ベースライン中: Beauty HR@10 1位、LastFM HR@10/NDCG@10 1位、ML-1M NDCG@10 1位タイ。
- 残差支配: HRLI@1 は教師より全DSで低減（残差が有害な DS ほど強く抑制）。

詳細: `V3_RESULTS.md`（生徒結果）, `PRELIM_ANALYSIS_RESULTS.md`（残差支配の事前分析: 混合率/HRLI）,
`KD_METHOD_SPEC.md`（蒸留の式・実装）, `FAILED_KD_METHODS.md`（不採用手法の記録）。

---

## ディレクトリ構成

```
seqKD/
├── src/
│   ├── main.py          # 学習/評価エントリ
│   ├── utils.py         # 引数 + EarlyStopping
│   ├── trainers.py      # Trainer 群（PL は AdaptiveRankingCompTrainer, gate_fixed=1.0）
│   ├── dataset.py, metrics.py
│   ├── run_loss_ablation.py   # KL/PL/RD の損失 ablation ランナー
│   └── model/
│       ├── bsarec.py          # 教師
│       └── kd_student_v3.py   # 提案生徒 FreqMamba
└── archive/             # 旧・不採用の実験資産（v1/v2, KL grid, HS-KD/CDD/補完/RD, 診断, 旧結果）
```
データ・チェックポイント・ログ（`src/data/`, `src/output/`, `*.pt/*.log`）は容量のため Git 管理外。
データは BSARec / FMLP-Rec の元リポジトリから取得し `../../BSARec/src/data/` に配置。

## 実行

作業ディレクトリは `seqKD/src/`。

### 提案（FreqMamba + PL 蒸留）
```bash
python main.py \
  --model_type kdstudent_v3 --do_distill \
  --kd_mode adaptive_rank_comp --gate_fixed 1.0 --rank_k 10 \
  --teacher_type bsarec --teacher_ckpt ../../BSARec/src/output/BSARec_<DS>_grid_*.pt \
  --teacher_num_attention_heads <h> --teacher_alpha <a> --teacher_c <c> \
  --alpha <a> --c <c> --d_state 16 --d_conv 4 --expand 1 \
  --hidden_size 64 --num_hidden_layers 2 \
  --hidden_dropout_prob <d> --attention_probs_dropout_prob <d> \
  --lr 0.001 --batch_size 256 --epochs 200 --patience 10 \
  --data_name <DS> --train_name fmamba_pl_<DS>
```
データセット別: Beauty `a0.7 c5 d0.5 h2` / LastFM `a0.9 c3 d0.5 h1` / ML-1M `a0.3 c9 d0.2 h1`。

### 蒸留なし（L_rec のみ, 床）
```bash
python main.py --model_type kdstudent_v3 --data_name <DS> --train_name fmamba_noKD_<DS> \
  --alpha <a> --c <c> --d_state 16 --d_conv 4 --expand 1 --hidden_size 64 --num_hidden_layers 2 \
  --hidden_dropout_prob <d> --attention_probs_dropout_prob <d> --lr 0.001 --epochs 200 --patience 10
```

### 損失 ablation の一括実行・集計
```bash
python run_loss_ablation.py --datasets ML-1M Beauty LastFM --skip_existing
python run_loss_ablation.py --report_only --datasets ML-1M Beauty LastFM
```

### 評価のみ
```bash
python main.py --do_eval --load_model <train_name> --model_type kdstudent_v3 --data_name <DS> \
  --alpha <a> --c <c> --d_state 16 --d_conv 4 --expand 1
```

## 環境
`mamba-ssm` / `causal-conv1d`（CUDA）が必要（Mamba 枝）。BSARec の conda 環境を利用。
