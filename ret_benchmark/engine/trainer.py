# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import datetime
import time

import numpy as np
import torch
import os

from ret_benchmark.data.evaluations.eval import AccuracyCalculator
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.metric_logger import MetricLogger
from ret_benchmark.utils.log_info import log_info
from ret_benchmark.modeling.xbm import XBM
from ret_benchmark.data.samplers import ImportanceSampler


def flush_log(writer, iteration):
    for k, v in log_info.items():
        if isinstance(v, np.ndarray):
            writer.add_histogram(k, v, iteration)
        else:
            writer.add_scalar(k, v, iteration)
    for k in list(log_info.keys()):
        del log_info[k]


@torch.no_grad()
def compute_all_feats(cfg, model, train_loader, xbm):
    left_old_feats = xbm.ptr #- cfg.DATA.TRAIN_BATCHSIZE
    right_old_feats = xbm.ptr + cfg.DATA.TRAIN_BATCHSIZE

    num_samples = xbm.K if xbm.is_full else left_old_feats
    xbm_feats = torch.zeros(num_samples, 128).cuda()
    xbm_targets = torch.zeros(num_samples, dtype=torch.long).cuda()

    # prev_tbs = cfg.DATA.TRAIN_BATCHSIZE
    # cfg.DATA.TRAIN_BATCHSIZE = 256
    # train_loader = build_data(cfg, is_train=True)
    _train_loader = iter(train_loader)
    model.eval()
    counter = 0
    while counter < num_samples:
        try:
            images, targets, _ = _train_loader.next()
        except StopIteration:
            _train_loader = iter(train_loader)
            images, targets, _ = _train_loader.next()

        feats = model(images.cuda())

        if counter + len(images) > num_samples:
            xbm_feats[counter:] = feats[:num_samples - counter]
            xbm_targets[counter:] = targets[:num_samples - counter].cuda()
        else:
            xbm_feats[counter: counter + len(images)] = feats
            xbm_targets[counter: counter + len(images)] = targets.cuda()
        counter += len(images)
    model.train()

    # cfg.DATA.TRAIN_BATCHSIZE = prev_tbs
    if left_old_feats >= 0:
        xbm.feats[:left_old_feats] = xbm_feats[:left_old_feats]
        xbm.targets[:left_old_feats] = xbm_targets[:left_old_feats]
        if xbm.is_full:
            xbm.feats[right_old_feats:] = xbm_feats[right_old_feats:]
            xbm.targets[right_old_feats:] = xbm_targets[right_old_feats:]
    else:
        xbm.feats[right_old_feats: left_old_feats] = xbm_feats[right_old_feats: left_old_feats]
        xbm.targets[right_old_feats: left_old_feats] = xbm_targets[right_old_feats: left_old_feats]


def do_train(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    checkpointer,
    writer,
    device,
    checkpoint_period,
    arguments,
    logger,
):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITERS

    best_iteration = -1
    best_mapr = 0

    start_training_time = time.time()
    end = time.time()

    if cfg.XBM.ENABLE:
        logger.info(">>> use XBM")
        xbm = XBM(cfg)

    iteration = 0
    _train_loader = iter(train_loader)
    while iteration <= max_iter:
        if cfg.DATA.SAMPLE == "ImportanceSampler" and iteration > cfg.XBM.START_ITERATION:
            train_loader.batch_sampler.update_scorer(model)
        try:
            images, targets, indices = _train_loader.next()
        except StopIteration:
            _train_loader = iter(train_loader)
            images, targets, indices = _train_loader.next()

        if (
            iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter
        ) and iteration > 0:
            model.eval()
            logger.info("Validation")

            labels = val_loader[0].dataset.label_list
            labels = np.array([int(k) for k in labels])
            feats = feat_extractor(model, val_loader[0], logger=logger)
            ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision"), exclude=())
            ret_metric = ret_metric.get_accuracy(feats, feats, labels, labels, True)
            mapr_curr = ret_metric['mean_average_precision_at_r']
            for k, v in ret_metric.items():
                log_info[f"e_{k}"] = v

            scheduler.step(log_info[f"e_precision_at_1"])
            log_info["lr"] = optimizer.param_groups[0]["lr"]
            if mapr_curr > best_mapr:
                best_mapr = mapr_curr
                best_iteration = iteration
                logger.info(f"Best iteration {iteration}: {ret_metric}")
            else:
                logger.info(f"Performance at iteration {iteration:06d}: {ret_metric}")
            flush_log(writer, iteration)

        model.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        feats = model(images)

        # if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
        #     xbm.enqueue_dequeue(feats.detach(), targets.detach())

        loss = criterion(feats, targets, feats, targets)
        log_info["batch_loss"] = loss.item()

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION and not xbm.is_empty:
            if cfg.XBM.UPDATE_FEATS_ITERATION > 0 and \
               iteration % cfg.XBM.UPDATE_FEATS_ITERATION == 0 and \
               iteration > cfg.XBM.UPDATE_FEATS_START_ITERATION:
                t0 = time.time()
                compute_all_feats(cfg, model, train_loader, xbm)
                print(f"Update all feats in XBM: {time.time() - t0}s")
                if iteration >= 10000 and iteration % 1000 == 0:
                    os.makedirs(writer.log_dir, exist_ok=True)
                    np.save(writer.log_dir + f"/xbm_pos_freqs_{iteration:06d}.npy", criterion.total_pos_freqs,
                            allow_pickle=True)
                    np.save(writer.log_dir + f"/xbm_neg_freqs_{iteration:06d}.npy", criterion.total_neg_freqs,
                            allow_pickle=True)
            elif iteration >= 30000 and iteration % 1000 == 0:
                os.makedirs(writer.log_dir, exist_ok=True)
                np.save(writer.log_dir + f"/xbm_pos_freqs_{iteration:06d}.npy", criterion.total_pos_freqs,
                        allow_pickle=True)
                np.save(writer.log_dir + f"/xbm_neg_freqs_{iteration:06d}.npy", criterion.total_neg_freqs,
                        allow_pickle=True)

            xbm_feats, xbm_targets = xbm.get()
            xbm_loss = criterion(feats, targets, xbm_feats, xbm_targets)
            log_info["xbm_loss"] = xbm_loss.item()
            loss = 1.5 * loss + cfg.XBM.WEIGHT * xbm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION - 1: # -1 to collect xbm of batch_size
            xbm.enqueue_dequeue(feats.detach(), targets.detach())

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 40 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.1f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

            log_info["loss"] = loss.item()
            flush_log(writer, iteration)

        if iteration % checkpoint_period == 0 and cfg.SAVE:
            checkpointer.save("model_{:06d}".format(iteration))
            pass

        del feats
        del loss

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    logger.info(f"Best iteration: {best_iteration :06d} | best MAP@R {best_mapr} ")
    writer.close()
