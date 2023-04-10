import collections
import json
import time
import copy

import gc
import inspect
import numpy as np
import torch
import torch.utils.data

from torch import cuda
from pathlib import Path
from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module

open_set_config = {
    "OfficeHome": 50
}


def free_memory(to_delete: list):
    calling_namespace = inspect.currentframe().f_back

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()
        
def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(
    test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None
):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    
    # [#] open-set config
    
    if args.open_set:
        print("Setting up open-set datasets ...")
        open_target = (test_envs[0], open_set_config[args.dataset])
    else:
        open_target = None
        
    # [#] ================
    
    dataset, in_splits, out_splits = get_dataset(
        test_envs, args, hparams, algorithm_class, open_target
    )
    hparams["class_names"] = dataset.class_names
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(
            testenv_name, testenv_name.replace(".", "")
        )
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(
        f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})"
    )

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(
        f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}"
    )

    # setup loaders
    # [#] changing loaders for open-set
    if test_envs[0] == -1:
        train_loaders = [
            InfiniteDataLoader(
                dataset=env,
                weights=env_weights,
                batch_size=batch_size,
                num_workers=dataset.N_WORKERS,
            )
            for (env, env_weights), batch_size in zip(in_splits, batch_sizes)
        ]
    else:
        train_loaders = [
            InfiniteDataLoader(
                dataset=env,
                weights=env_weights,
                batch_size=batch_size,
                num_workers=dataset.N_WORKERS,
            )
            for (env, env_weights), batch_size in iterator.train(
                zip(in_splits, batch_sizes)
            )
        ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {
            "dataset": env,
            "batch_size": batchsize,
            "num_workers": dataset.N_WORKERS,
        }
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    # if "DFC" in args.algorithm and args.algorithm.split("_")[1] != "STAGE3":
    if "DFC" in args.algorithm:
        clip_model = algorithms.CLIP(hparams)
        algorithm = algorithm_class(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset) - len(test_envs),
            hparams,
            clip_model,
        )
    else:
        algorithm = algorithm_class(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset) - len(test_envs),
            hparams,
        )
    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #[#] init. from stage 1 training
    if hparams["pretrained"] is not None and hparams["pretrained"] not in ["True", "False"]:
        # stage = hparams["pretrained"].split("/")[2].split("_")[0]
        print(hparams["pretrained"])
        # load stage 1 model for stage 2 training
        # if stage == "stage1":
        if "stage1" in hparams["pretrained"]:
            logger.info("Loading Stage 1 pre-trained model ...")
            model_path = hparams["pretrained"] + "TE{}_best.pth".format(test_envs[0])
            model_dict = torch.load(model_path)
            algorithm_dict = model_dict["model_dict"]
            algorithm.load_state_dict(algorithm_dict, strict=True)

        # load stage 2 model for stage 3 [MIRO+SWAD]
        # elif stage == "stage2":
        elif "stage2" in hparams["pretrained"]:
            logger.info("Loading Stage 2 pre-trained model ...")
            if args.open_set:
                model_path = hparams["pretrained"] + "TE{}_best.pth".format(args.model_load)
            else:
                model_path = hparams["pretrained"] + "TE{}_best.pth".format(test_envs[0])
            model_dict = torch.load(model_path)
            algorithm_dict = model_dict["model_dict"]

            # loading classifier
            if hparams["cls_pth"] != "":
                model_path_cls = hparams["cls_pth"] + "TE{}_best.pth".format(test_envs[0])
                model_dict_cls = torch.load(model_path_cls)
                algorithm_dict_cls = model_dict_cls["model_dict"]
                algorithm_dict["classifier.weight"] = (algorithm_dict["classifier.weight"] + algorithm_dict_cls["classifier.weight"]) / 2
                algorithm_dict["classifier.bias"] = (algorithm_dict["classifier.bias"] + algorithm_dict_cls["classifier.bias"]) / 2

            # # [#] modifying state dict for loading
            if "MIRO" in args.algorithm:
                new_state_dict = {}
                for name in algorithm_dict:
                    new_state_dict[name] = algorithm_dict[name]
                    if name.startswith("featurizer"):
                        new_state_dict["pre_" + name] = algorithm_dict[name]
                algorithm.load_state_dict(new_state_dict, strict=False)
            else:
                algorithm.load_state_dict(algorithm_dict, strict=False)
                
        else:
            logger.info("Loading LLT pre-trained model ...")
            model_path = hparams["pretrained"] + "TE{}_best.pth".format(test_envs[0])
            model_dict = torch.load(model_path)
            algorithm_dict = model_dict["model_dict"]
            algorithm.load_state_dict(algorithm_dict, strict=True)
            
                

    # [#] init. from MIRO + SWAD training
    elif args.miro_swad is not None and args.miro_swad != "None":
        logger.info("Loading MIRO+SWAD pre-trained model ...")
        model_path = args.miro_swad + "TE{}_best.pth".format(test_envs[0])
        model_dict = torch.load(model_path)
        algorithm_dict = model_dict["model_dict"]

        # [#] modifying state dict for loading
        new_state_dict = {}
        for name in algorithm_dict:
            if name.startswith("network.0"):
                new_state_dict["featurizer." + name[10:]] = algorithm_dict[name]
        algorithm.load_state_dict(new_state_dict, strict=False)
        
        
    # [#] init. from ERM training
    elif hparams["pretrained"] is not None and "erm" in hparams["pretrained"]:
        logger.info("Loading ERM pre-trained model ...")
        model_path = hparams["pretrained"] + "TE{}_best.pth".format(test_envs[0])
        model_dict = torch.load(model_path)
        algorithm_dict = model_dict["model_dict"]

        # [#] modifying state dict for loading
        new_state_dict = {}
        for name in algorithm_dict:
            if name.startswith("network.0"):
                new_state_dict["featurizer." + name[10:]] = algorithm_dict[name]
        algorithm.load_state_dict(new_state_dict, strict=False)

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
        classnames=hparams["class_names"]
    )

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, "LossValley")
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    # [#] acc. check for analysis
    if args.check_acc:
        if args.cls_pth:
            return evaluator.evaluate(algorithm)
        else:
            return evaluator.evaluate_cls(algorithm)
        
        
    # [#] ===========================
    # [#] eval code for sanity check
    # [#] ===========================
    
    if args.open_set and args.check_acc:
        accuracies, summaries = evaluator.evaluate(algorithm)
        results = {
                    "step": -1,
                    "epoch": -1,
                }
        results_keys = (
                list(summaries.keys())
                + sorted(accuracies.keys())
                + list(results.keys())
            )
        # merge results
        results.update(summaries)
        results.update(accuracies)

        # print
        if results_keys != last_results_keys:
            logger.info(misc.to_row(results_keys))
            last_results_keys = results_keys
        logger.info(misc.to_row([results[key] for key in results_keys]))
        records.append(copy.deepcopy(results))
        exit()
    
    # [#] ===========================
    
    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.cuda() for tensor in tensorlist]
            for key, tensorlist in batches.items()
        }

        inputs = {**batches, "step": step}

        # if "DFC" in args.algorithm and args.algorithm.split("_")[1] != "STAGE3":
        if "DFC" in args.algorithm:
            step_vals = algorithm.update(**inputs, clip_model=clip_model)
        else:
            step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = (
                list(summaries.keys())
                + sorted(accuracies.keys())
                + list(results.keys())
            )
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(
                    json.dumps(results, sort_keys=True, default=json_handler) + "\n"
                )

            checkpoint_vals = collections.defaultdict(lambda: [])

            # writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            # writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            # [#] Model saving
            temp_rec = Q(records)
            if (
                args.model_save
                and temp_rec.argmax("train_out")["train_out"]
                == temp_rec[-1]["train_out"]
                and step >= args.model_save
            ):
            # if (
            #     args.model_save
            #     and step >= args.model_save
            # ):
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_best.pth".format(test_env_str)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # if args.model_save and step >= args.model_save:
            #     ckpt_dir = args.out_dir / "checkpoints"
            #     ckpt_dir.mkdir(exist_ok=True)

            #     test_env_str = ",".join(map(str, test_envs))
            #     filename = "TE{}_{}.pth".format(test_env_str, step)
            #     if len(test_envs) > 1 and target_env is not None:
            #         train_env_str = ",".join(map(str, train_envs))
            #         filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
            #     path = ckpt_dir / filename

            #     save_dict = {
            #         "args": vars(args),
            #         "model_hparams": dict(hparams),
            #         "test_envs": test_envs,
            #         "model_dict": algorithm.cpu().state_dict(),
            #     }
            #     algorithm.cuda()
            #     if not args.debug:
            #         torch.save(save_dict, path)
            #     else:
            #         logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad:

                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row(
                        [results[key] for key in results_keys if key in results]
                    )
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm,
                    results["train_out"],
                    results["tr_outloss"],
                    prt_results_fn,
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                # del swad_algorithm.module
                # torch.cuda.empty_cache()s
                # free_memory([swad_algorithm])
                # swad_algorithm = swa_utils.AveragedModel(algorithm, rm_optimizer=True)  # reset
                if args.swad_fix:
                    if (step % 1000 == 0) and (step > 2500):
                        swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset
                else:
                    swad_algorithm = swa_utils.AveragedModel(algorithm)

        # if step % args.tb_freq == 0:
        # add step values only for tb log
        # writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

    # find best
    logger.info("---")
    records = Q(records)
    te_val_best = records.argmax("test_out")["test_in"]
    tr_val_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    in_key = "train_out"
    tr_val_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    # NOTE for clearity, report only training-domain validation results.
    ret = {
        #  "test-domain validation": te_val_best,
        "training-domain validation": tr_val_best,
        #  "last": last,
        #  "last (inD)": last_indomain,
        #  "training-domain validation (inD)": tr_val_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = (
            misc.to_row([results[key] for key in results_keys if key in results])
            + step_str
        )
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records
