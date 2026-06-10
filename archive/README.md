# archive/ — 旧・不採用の実験資産（削除はしていない）

採用構成は **生徒 = FreqMamba (KDStudent v3) / 蒸留 = Plackett-Luce listwise ランキング蒸留（KLより頑健）**。
それに関係しないファイルをここに退避した。戻す場合はファイルを元の場所に移し、`model/__init__.py` の import / MODEL_DICT を復元する。

## models/  — 旧モデル
- `kd_student.py` (v1, 純GRU生徒), `kd_student_v2.py` (v2, GLINT-RU風) … 旧生徒。
- `mlp_student.py`, `sigma.py`, `gru4rec.py`, `lrurec.py`, `fmlprec.py`, `duorec.py` … KD実験用に複製したベースライン（教師 BSARec 以外）。

## scripts/ — 旧スクリプト
- v1/v2 grid・ablation: `run_kdstudent.py`, `run_kdstudent_grid.py`, `run_v2_initial.py`, `run_ablation.py`, `run_ablation_v2.py`, `aggregate_*.py`
- v3 の KL+HS grid: `run_kdstudent_v3_grid.py`（v3アーキの結果は `../V3_RESULTS.md` に保存済）
- 不採用の蒸留: `run_hs_kd.py`(HS-KD), `run_cdd_initial.py`/`run_cdd_only.py`(CDD), `run_comp_experiment.py`(補完項), `run_kd.py`, `run_systematic_kd.py`
- 診断: `diagnose_adaptive_rank.py`(v1), `diagnose_adaptive_rank_v2.py`, `diagnose_complementarity.py`

## docs/ — 旧結果ドキュメント
- `KD_RESULTS.md`(v1), `V2_RESULTS.md`, `ABLATION_RESULTS.md`, `ABLATION_V2_RESULTS.md`,
  `hs_kd_report.md`, `kd_report.md`, `kdstudent_report.md`, `systematic_kd_report.md`

不採用の経緯は `../FAILED_KD_METHODS.md` に集約。
