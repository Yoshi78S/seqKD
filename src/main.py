import os
import time
import torch
import numpy as np

from model import MODEL_DICT
from trainers import (Trainer, DistillTrainer, HiddenStateDistillTrainer,
                      KDStudentDistillTrainer, AdaptiveRankingDistillTrainer,
                      AdaptiveRankingV2Trainer, AdaptiveRankingCompTrainer,
                      RankNaiveDistillTrainer)
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger, make_teacher_args
from dataset import get_seq_dic, get_dataloder, get_rating_matrix


def _build_model(model_type, args):
    return MODEL_DICT[model_type.lower()](args=args)


def main():
    args = parse_args()
    check_path(args.output_dir)
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # ----- data -----
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name + '_same_target.npy')

    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    logger.info(str(args))

    # ----- student model -----
    student = _build_model(args.model_type, args)
    logger.info(f"[Student] {args.model_type}")
    logger.info(student)
    n_params = sum(p.nelement() for p in student.parameters())
    logger.info(f"TIMING n_params {n_params}")

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(
        args.data_name, seq_dic, max_item
    )

    # ----- trainer (with or without distillation) -----
    if (args.do_distill or args.do_hs_distill) and not args.do_eval:
        if args.teacher_ckpt is None:
            raise ValueError("--teacher_ckpt is required when --do_distill/--do_hs_distill is set.")
        teacher_args = make_teacher_args(args)
        logger.info(f"[Teacher args] type={args.teacher_type}, "
                    f"hidden_size={teacher_args.hidden_size}, "
                    f"num_hidden_layers={teacher_args.num_hidden_layers}")
        teacher = _build_model(args.teacher_type, teacher_args)
        state = torch.load(args.teacher_ckpt,
                           map_location="cuda" if args.cuda_condition else "cpu")
        teacher.load_state_dict(state)
        logger.info(f"[Teacher] {args.teacher_type} loaded from {args.teacher_ckpt}")

        if getattr(args, 'kd_mode', 'kl') == 'rank_naive':
            # RD-style naive pointwise ranking distillation.
            trainer = RankNaiveDistillTrainer(
                student, teacher,
                train_dataloader, eval_dataloader, test_dataloader,
                args, logger,
            )
        elif getattr(args, 'kd_mode', 'kl') == 'adaptive_rank_comp':
            # Complementary distillation: main PL on z_ord + rho-gated complement.
            trainer = AdaptiveRankingCompTrainer(
                student, teacher,
                train_dataloader, eval_dataloader, test_dataloader,
                args, logger,
            )
        elif getattr(args, 'kd_mode', 'kl') == 'adaptive_rank_v2':
            # Adaptive ranking distillation v2 (pre/post-residual interpolation).
            trainer = AdaptiveRankingV2Trainer(
                student, teacher,
                train_dataloader, eval_dataloader, test_dataloader,
                args, logger,
            )
        elif getattr(args, 'kd_mode', 'kl') == 'adaptive_rank':
            # Adaptive ranking distillation v1 (attention-temperature, deprecated).
            trainer = AdaptiveRankingDistillTrainer(
                student, teacher,
                train_dataloader, eval_dataloader, test_dataloader,
                args, logger,
            )
        elif args.do_hs_distill:
            if args.model_type.lower() in ('kdstudent', 'kdstudent_v2', 'kdstudent_v3'):
                trainer = KDStudentDistillTrainer(
                    student, teacher,
                    train_dataloader, eval_dataloader, test_dataloader,
                    args, logger,
                )
            else:
                trainer = HiddenStateDistillTrainer(
                    student, teacher,
                    train_dataloader, eval_dataloader, test_dataloader,
                    args, logger,
                )
        else:
            trainer = DistillTrainer(
                student, teacher,
                train_dataloader, eval_dataloader, test_dataloader,
                args, logger,
            )
    else:
        trainer = Trainer(
            student,
            train_dataloader, eval_dataloader, test_dataloader,
            args, logger,
        )

    # ----- eval-only path -----
    if args.do_eval:
        if args.load_model is None:
            logger.info("No model input!")
            return
        ckpt = os.path.join(args.output_dir, args.load_model + '.pt')
        trainer.load(ckpt)
        logger.info(f"Load model from {ckpt} for test!")
        scores, result_info = trainer.test(0)
        logger.info(args.train_name)
        logger.info(result_info)
        return

    # ----- training -----
    early_stopping = EarlyStopping(args.checkpoint_path, logger=logger,
                                   patience=args.patience, verbose=True)
    wall_start = time.perf_counter()
    completed_epochs = 0
    for epoch in range(args.epochs):
        trainer.train(epoch)
        scores, _ = trainer.valid(epoch)
        completed_epochs = epoch + 1
        early_stopping(np.array(scores[-1:]), trainer.model, epoch=epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    wall_elapsed = time.perf_counter() - wall_start
    logger.info(f"TIMING wall_train {wall_elapsed:.4f}s")
    logger.info(f"TIMING epochs_run {completed_epochs}")

    train_times = trainer.train_epoch_times
    if train_times:
        logger.info(f"TIMING train_epoch_mean {np.mean(train_times):.4f}s")
        logger.info(f"TIMING train_epoch_sum {np.sum(train_times):.4f}s")
    eval_times = trainer.eval_epoch_times
    if eval_times:
        logger.info(f"TIMING valid_epoch_mean {np.mean(eval_times):.4f}s")

    logger.info("---------------Test Score---------------")
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    best_epoch = early_stopping.best_epoch if early_stopping.best_epoch is not None else 0
    logger.info(f"Loaded best-val checkpoint from epoch {best_epoch}")
    scores, result_info = trainer.test(best_epoch)
    logger.info(args.train_name)
    logger.info(result_info)


if __name__ == "__main__":
    main()
