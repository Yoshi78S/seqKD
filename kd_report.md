# seqKD — 知識蒸留 Phase 1 実験レポート

## 概要

SimRec (WWW 2023) の逐次推薦版として、Mamba ベースの教師 (SIGMA) から軽量 MLP 生徒への **prediction-level 知識蒸留**を行う Phase 1 実験。

**目的**: 教師の推論品質を軽量な MLP に転写し、推論速度と精度のトレードオフを改善できるか検証する。

## アーキテクチャ

### 教師 (Teacher): SIGMA

BSARec ベンチマークで訓練済みのチェックポイントを凍結して使用 (`SIGMA_{Dataset}_bench.pt`)。

| 項目 | 値 |
|---|---|
| ベース | GMambaBlock (順方向 Mamba + 逆方向 Mamba + GRU の 3 ストリーム) |
| num_hidden_layers | 1 |
| hidden_size | 64 |
| hidden_dropout_prob | 0.2 |
| d_state / d_conv / expand | 32 / 4 / 2 |
| 訓練時の状態 | **完全に凍結** (`requires_grad=False`, `eval()` モード) |

教師の推論は各 batch で `torch.no_grad()` の下で実行。勾配は生徒側のみ。

### 生徒 (Student): MLPStudentModel

([seqKD/src/model/mlp_student.py](src/model/mlp_student.py))

```
input_ids
   ↓
item_embeddings + position_embeddings
   ↓
LayerNorm → Dropout
   ↓
FeedForward ×N  (位置独立 MLP、系列混合なし)
   ↓
[batch, seq_len, hidden]  →  last position  →  dot(item_embeddings.weight)  →  logits
```

| 項目 | 値 |
|---|---|
| エンコーダ | `FeedForward` ブロック (Linear→GELU→Linear→Dropout→LayerNorm+residual) × N |
| num_hidden_layers | 2 |
| hidden_size | 64 |
| hidden_dropout_prob | 0.5 |
| inner_size | 256 (= 4 × hidden_size) |

**設計意図**: Attention / FFT / Mamba 等の**系列混合操作を一切排除**。位置情報は position embedding に押し込み、各位置を独立に MLP で変換するだけ。これにより:
- 推論が軽い (O(L×d²) — 系列長に線形)
- 短系列データで過学習しにくい (Mamba/SSM が苦手な領域で安定)
- 教師の knowledge が唯一の「系列間関係」情報源 → KD の効果を純粋に評価できる

## 損失設計

```
L = L_rec + λ_kd × L_KD
```

### L_rec: 推薦損失 (生徒の自律学習)

```python
L_rec = CrossEntropy(student_logits, answers)
```

- 生徒の最終位置の hidden state から全アイテム vocab に対する logits を計算
- 正解アイテム `answers` に対する標準 CE
- 教師非依存 — 生徒が独力で推薦を学ぶ部分

### L_KD: 知識蒸留損失 (SimRec 風 soft-label KD)

```python
log_p_s = log_softmax(student_logits / T, dim=-1)
p_t     = softmax(teacher_logits / T, dim=-1)

L_KD = KL(p_t || log_p_s) × T²
```

- **温度 T=2.0**: 教師の logits を "ソフト化" して確率分布を平滑化。
  T が大きいほど教師の "dark knowledge" (2位以下のアイテムの相対順序情報) が生徒に伝わりやすい
- **T² スケーリング**: KL divergence の勾配が T に反比例するのを補正 (Hinton 2015 の標準手法)
- **全アイテム vocab 上で計算**: サンプリングなし、teacher と student の logits を完全に比較

### ハイパラ

| パラメータ | 値 | 説明 |
|---|---|---|
| λ_kd | 1.0 | L_rec と L_KD の重み (等重) |
| T (kd_temperature) | 2.0 | ソフト化温度 |

## 実装詳細

([seqKD/src/trainers.py](src/trainers.py))

### DistillTrainer (Trainer を継承)

```python
class DistillTrainer(Trainer):
    def _teacher_logits(self, input_ids):
        with torch.no_grad():
            seq_out = self.teacher(input_ids)[:, -1, :]
            return seq_out @ self.teacher.item_embeddings.weight.T

    def _student_logits(self, input_ids):
        seq_out = self.model(input_ids)[:, -1, :]
        return seq_out @ self.model.item_embeddings.weight.T

    def _compute_train_loss(self, batch):
        _, input_ids, answers, _, _ = batch
        student_logits = self._student_logits(input_ids)
        l_rec = F.cross_entropy(student_logits, answers)

        teacher_logits = self._teacher_logits(input_ids)
        T = self.kd_temperature
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)
        l_kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

        return l_rec + self.lambda_kd * l_kd
```

**ポイント**:
- Trainer の `_compute_train_loss` を override するだけの最小差分設計
- 評価 (valid/test) は生徒単独 — BSARec と同一の full-sort 評価プロトコル
- 教師の `item_embeddings` と生徒の `item_embeddings` は**別物** (重みを共有していない)

### 教師アーキテクチャの復元

教師の checkpoint は state_dict のみ。アーキテクチャを復元するため `make_teacher_args()` で CLI からオーバーライド:

```bash
--teacher_type sigma \
--teacher_ckpt ../../BSARec/src/output/SIGMA_Beauty_bench.pt \
--teacher_num_hidden_layers 1 \
--teacher_hidden_dropout_prob 0.2 \
--teacher_attention_probs_dropout_prob 0.2
```

`make_teacher_args()` は student args のコピーを作り、`teacher_*` フラグで指定された値だけ上書き。データ形状 (item_size, max_seq_length 等) は共有。

## 実験結果

3 データセット (LastFM, Beauty, ML-1M) で実行済み。

### パラメータ数

| Dataset | 生徒 (MLP) | 教師 (SIGMA, frozen) | 比率 |
|---|---:|---:|---:|
| LastFM | 303,168 | 366,595 | 0.83× |
| Beauty | 844,288 | 907,715 | 0.93× |
| ML-1M  | 288,448 | 351,875 | 0.82× |

> 生徒は教師の ~83-93% のパラメータ数。差分は主に Mamba / GRU / Conv1d の追加パラメータ。

### 学習設定

| Dataset | Best Epoch | Epochs Run | Wall Train | Train/Epoch | Infer. (s) |
|---|---:|---:|---:|---:|---:|
| LastFM | 18 | 29 | 35s | 1.01s | 0.18 |
| Beauty | ~24 | 35 | 285s | 5.10s | 3.49 |
| ML-1M  | 15 | 26 | 242s | 8.79s | 0.52 |

### テストスコア (best-val チェックポイント)

| Dataset | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| LastFM | 0.0413 | 0.0275 | 0.0495 | 0.0301 | 0.0679 | 0.0348 |
| Beauty | 0.0593 | 0.0418 | 0.0825 | 0.0493 | 0.1146 | 0.0573 |
| ML-1M  | 0.1303 | 0.0909 | 0.1919 | 0.1107 | 0.2727 | 0.1311 |

## 教師 (SIGMA) との比較

| Dataset | MLP+KD HR@10 | SIGMA HR@10 | 差分 | 解釈 |
|---|---:|---:|---:|---|
| **LastFM** | **0.0495** | 0.0404 | **+22.5%** | 生徒が教師を超えた |
| **Beauty** | **0.0825** | 0.0707 | **+16.7%** | 生徒が教師を超えた |
| ML-1M  | 0.1919 | **0.2858** | **−32.9%** | 生徒が大きく劣る |

### 生徒が教師を超えるメカニズム (LastFM / Beauty)

1. **SIGMA の弱点**: SIGMA は小規模・短系列データで精度が低い (LastFM で全 13 モデル中最下位)。3 ストリーム (Mamba 順/逆 + GRU) の合成が少ないサンプルで不安定
2. **KD の正則化効果**: 教師の soft label が生徒にとって暗黙の正則化として働く。「教師が少し推している 2 位・3 位のアイテム」の情報が生徒の汎化を助ける
3. **MLP の安定性**: 系列混合操作がないため、短系列でのノイズ増幅がなく、position embedding + KD だけで十分な精度を達成

### 生徒が教師に劣るメカニズム (ML-1M)

1. **ML-1M の系列長 (avg=165)**: 系列混合なしの MLP では遠距離のアイテム依存関係を捉えられない
2. **SIGMA の強み**: Mamba の状態空間モデルが長系列を効率的に処理。ML-1M で全モデル中最強 (HR@10=0.286)
3. **KD の限界**: prediction-level KD は logits の分布を mimicking するだけで、教師の内部表現 (系列間の依存関係パターン) は直接転写されない

## ベンチマーク全体における位置付け

| Dataset | MLP+KD HR@10 | ベンチ最強 | BERT4Rec | SASRec | FMLPRec | GRU4Rec |
|---|---:|---:|---:|---:|---:|---:|
| LastFM | 0.0495 | 0.0670 (BSARec) | 0.0486 | 0.0560 | 0.0587 | 0.0422 |
| Beauty | 0.0825 | 0.0975 (BSARec) | 0.0693 | 0.0496 | 0.0532 | 0.0267 |
| ML-1M  | 0.1919 | 0.2858 (SIGMA) | 0.2371 | 0.2094 | 0.2142 | 0.1834 |

**MLP+KD vs 既存ベースライン**:
- **Beauty**: MLP+KD (0.0825) > BERT4Rec (0.0693), SASRec (0.0496), FMLPRec (0.0532) — **MLP+KD が Transformer 系を超える**
- **LastFM**: MLP+KD (0.0495) ≈ BERT4Rec (0.0486) — ほぼ同等
- **ML-1M**: MLP+KD (0.1919) < BERT4Rec (0.2371) — 長系列では不足

## 推論時間の比較

| Dataset | MLP+KD | SIGMA (教師) | BSARec | FMLPRec | GRU4Rec |
|---|---:|---:|---:|---:|---:|
| LastFM | **0.18s** | 0.20s | 0.25s | 0.21s | 0.26s |
| Beauty | 3.49s | 3.64s | 4.95s | **3.41s** | 4.43s |
| ML-1M  | **0.52s** | 0.77s | 0.78s | 0.65s | 0.64s |

MLP 生徒は**推論時間で教師 SIGMA と同等かそれ以下**。ML-1M では SIGMA の 68% の推論時間で達成。ただし精度差が大きいため、単純な置き換えには使えない。

## 未実施 (Phase 2/3 予定)

README に記載の将来構想:
- **Phase 2**: 系列長適応的な重み付け — 長系列 / 短系列でλ_kd を動的に変える
- **Phase 3**: 教師の信頼度ベーススケーリング — 教師が自信のある予測ほど強く蒸留

## 残り実験

| Dataset | MLP+KD |
|---|---|
| LastFM | ✅ |
| Beauty | ✅ |
| ML-1M | ✅ |
| Sports_and_Outdoors | ❌ (SIGMA checkpoint 完走後に実行可能) |
| Toys_and_Games | ❌ |
| Yelp | ❌ |

`bash run_all.sh` で Phase 1 → Phase 2 連続実行可能 (SIGMA checkpoint が揃ったデータセットから自動的に KD 実行)。

## ファイル構成

```
seqKD/
├── README.md                         # 実験概要 + Quick start
├── kd_report.md                      # 本レポート
├── scripts/
│   ├── run_kd_lastfm.sh              # 単発実行スクリプト
│   └── run_kd_ml-1m.sh
└── src/
    ├── main.py                       # 学習エントリポイント (--do_distill で KD 有効化)
    ├── trainers.py                   # Trainer + DistillTrainer
    ├── dataset.py                    # BSARec と同一
    ├── metrics.py                    # BSARec と同一
    ├── utils.py                      # parse_args() に KD 用引数追加 + make_teacher_args()
    ├── run_kd.py                     # 全データセット一括実行ランナー
    ├── output/                       # ログ + チェックポイント
    │   ├── MLP_LastFM_kd.{log,pt}
    │   ├── MLP_Beauty_kd.{log,pt}
    │   └── MLP_ML-1M_kd.{log,pt}
    └── model/
        ├── __init__.py               # MODEL_DICT: mlp_student, sigma
        ├── _abstract_model.py        # BSARec と同一
        ├── _modules.py               # BSARec と同一 (FeedForward 含む)
        ├── mlp_student.py            # MLP 生徒モデル
        └── sigma.py                  # SIGMA 教師モデル (BSARec と同一)
```
