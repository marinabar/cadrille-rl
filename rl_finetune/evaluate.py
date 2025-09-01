from train_cadrille_grpo_base import TrainConfig, collate_img_pc_v1, get_reward_function, optimize_model_memory, setup, cleanup
from utils_cadrille import get_metrics_from_texts, init_pool
from tqdm import tqdm
import pyrallis
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader

def evaluate_model_mm(config, model, processor, eval_examples, device, collate_fn, batch_size=8, normalize="fixed"):

    model.eval()
    print("\n" + "=" * 50)
    print("EVALUATION ON", len(eval_examples), "EXAMPLES")
    print("=" * 50)

    dataloader = DataLoader(eval_examples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    ious, cds, aucs, cos_sims = [], [], [], []
    n_incorrect, n_failed_intersect = 0, 0

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            generated_ids = model.generate(input_ids=batch['input_ids'].to(model.device),
                                           attention_mask=batch['attention_mask'].to(model.device),
                                           point_clouds=batch['point_clouds'].to(model.device),
                                           is_pc=batch['is_pc'].to(model.device),
                                           is_img=batch['is_img'].to(model.device),
                                           pixel_values_videos=batch['pixel_values_videos'].to(
                                               model.device) if batch.get('pixel_values_videos',
                                                                          None) is not None else None,
                                           video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get(
                                               'video_grid_thw', None) is not None else None,
                                           max_new_tokens=768,
                                           bad_words_ids=[[model.config.video_token_id]],
                                           temperature=config.temperature,
                                           do_sample=True,
                                           top_p=config.top_p,
                                           top_k=50,
                                           )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            decoded_texts = py_strings
            print("decoded_texts : ", decoded_texts)
            pred_metrics = get_metrics_from_texts(decoded_texts, batch["mesh_path"], max_workers=24, normalize=normalize)
            for metrics in pred_metrics:
                if metrics is None or metrics["iou"] is None or metrics["cd"] is None:
                    n_incorrect += 1
                    continue
                if metrics["iou"] < 0:
                    n_failed_intersect += 1
                else:
                    ious.append(metrics["iou"])
                    cds.append(metrics["cd"])
                    #aucs.append(metrics["auc"])
                    #cos_sims.append(metrics["mean_cos"])


    print(f"IoU mean {np.mean(ious)}, median {np.median(ious)}")
    print(f"CD mean {np.mean(cds)}, median {np.median(cds)}")
    print(f"Invalid generations fraction: {n_incorrect / len(eval_examples)}")
    print(f"Intersect failure fraction: {n_failed_intersect / len(eval_examples)}")
    print("=" * 50)

    model.train()
    return ious, cds, n_incorrect / len(eval_examples), n_failed_intersect / len(eval_examples)


def evaluate_model_mm2(config, model, processor, eval_examples, device, collate_fn, batch_size=8):
    from old_utils import extract_mesh_from_texts
    model.eval()
    print("\n" + "=" * 50)
    print("EVALUATION ON", len(eval_examples), "EXAMPLES")
    print("=" * 50)

    dataloader = DataLoader(eval_examples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=10)
    ious, cds = [], []
    n_incorrect, n_failed_intersect = 0, 0

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            print(batch.keys())
            generated_ids = model.generate(input_ids=batch['input_ids'].to(model.device),
                                           attention_mask=batch['attention_mask'].to(model.device),
                                           point_clouds=batch['point_clouds'].to(model.device),
                                           is_pc=batch['is_pc'].to(model.device),
                                           is_img=batch['is_img'].to(model.device),
                                           pixel_values_videos=batch['pixel_values_videos'].to(
                                               model.device) if batch.get('pixel_values_videos',
                                                                          None) is not None else None,
                                           video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get(
                                               'video_grid_thw', None) is not None else None,
                                           max_new_tokens=768,
                                           bad_words_ids=[[model.config.video_token_id]],
                                           temperature=config.temperature,
                                           do_sample=config.do_sample,
                                           top_p=config.top_p,
                                           top_k=50,
                                          )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
            ]
            py_strings = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            decoded_texts = py_strings
            results = extract_mesh_from_texts(decoded_texts, batch["mesh_path"])
            # print("IOUS:", pred_ious, flush=True)
            # for i, py_string in enumerate(decoded_texts):
            for i, res in enumerate(results):
                if res is None or res["iou"] is None:
                    n_incorrect += 1
                    continue
                pred_iou = res["iou"]
                pred_cd = res["cd"]
                if pred_iou < 0:
                    n_failed_intersect += 1
                else:
                    ious.append(pred_iou)
                    cds.append(pred_cd)


    print(f"IoU mean {np.mean(ious)}, median {np.median(ious)}")
    print(f"CD mean {np.mean(cds)}, median {np.median(cds)}")
    print(f"Invalid generations fraction: {n_incorrect / len(eval_examples)}")
    print(f"Intersect failure fraction: {n_failed_intersect / len(eval_examples)}")
    print("=" * 50)

    model.train()
    return ious, cds, n_incorrect / len(eval_examples), n_failed_intersect / len(eval_examples)


@pyrallis.wrap()
def main(config: TrainConfig):

    from cad_recode_model_mm import Cadrille

    from transformers import AutoProcessor
    from dataset_utils import RealDatasetMM

    from comet_ml import start

    print("Starting evaluation")
    experiment = start(
        api_key="CfQGtyWGF13CZEsUvXBeuPaSf",
        project_name="cad",
        workspace="marinabar",
    )
    experiment.set_name(config.name)
    params = {k: getattr(config, k) for k in config.__annotations__}
    experiment.log_parameters(params)

    rank = 0
    torch.cuda.set_device(rank)

    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None

    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=rank).train().to(device = torch.device(f"cuda:{rank}"))

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                              min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28,
                                              padding_side="left",
                                              )

    #eval_data_deepcad = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_test', file_name='test.pkl', n_points=256, size=1000)
    eval_data_fusion = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/fusion360_test', file_name='test.pkl', n_points=256, size=1000)
    eval_data_deepcad = RealDatasetMM(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train', file_name=config.train_file, n_points=256, mode=config.train_mode, noise_scale_pc=0.01, size=1000)

    print(f"Rank {rank}: Initializing datasets")

    model = optimize_model_memory(model)

    collate_fn = partial(
        collate_img_pc_v1,
        processor=processor,
        n_points=256)

    print("Initializing worker pool")
    init_pool(20)


    """
    overfit_dataset.mode = "img"

    ious_img, cds_img, incorrect_img, failed_intersect_img = evaluate_model_mm(model, processor, overfit_dataset, 0,
                                                            collate_fn,
                                                            batch_size=500)
    
    overfit_dataset.mode = "pc"

    ious_pc, cds_pc, incorrect_pc, failed_intersect_pc = evaluate_model_mm(model, processor, overfit_dataset, 0,
                                                            collate_fn,
                                                            batch_size=500)                    
    experiment.log_metrics({

        # Point‐cloud Fusion360
        "eval/pc/train_500 test/IoU mean":   np.mean(ious_pc),
        "eval/pc/train_500 test/CD mean":    np.mean(cds_pc),
        "eval/pc/train_500 test/IoU median": np.median(ious_pc),
        "eval/pc/train_500 test/CD median":  np.median(cds_pc),

        # Image DeepCAD
        "eval/img/train_500 test/IoU mean":   np.mean(ious_img),
        "eval/img/train_500 test/CD mean":    np.mean(cds_img),
        "eval/img/train_500 test/IoU median": np.median(ious_img),
        "eval/img/train_500 test/CD median":  np.median(cds_img),
        })
    """
    """
    eval_data_deepcad.mode = 'pc'
    eval_data_fusion.mode = 'pc'
    ious, cds,  aucs, cos_sims, incorrect, failed_intersect = evaluate_model_mm2(model, processor, eval_data_deepcad, 0,
                                                            collate_fn,
                                                            batch_size=500)
    
    ious_f, cds_f, aucs_f, cos_sims_f, incorrect_f, failed_intersect_f = evaluate_model_mm2(model, processor, eval_data_fusion,
                                                                    0, collate_fn,
                                                                    batch_size=500)

    eval_data_deepcad.mode = 'img'
    eval_data_fusion.mode = 'img'
    ious_img, cds_img, aucs_img, cos_sims_img, incorrect_img, failed_intersect_img = evaluate_model_mm2(model, processor, eval_data_deepcad, 0,
                                                            collate_fn,
                                                            batch_size=500)
    
    ious_f_img, cds_f_img, aucs_f_img, cos_sims_f_img, incorrect_f_img, failed_intersect_f_img = evaluate_model_mm2(model, processor, eval_data_fusion,
                                                                    0, collate_fn,
                                                                    batch_size=500)
    """

    eval_data_deepcad.mode = 'img'
    eval_data_fusion.mode = 'img'
    ious_img, cds_img, incorrect_img, failed_intersect_img = evaluate_model_mm(config, model, processor, eval_data_deepcad, 0,
                                                            collate_fn,
                                                            batch_size=500, normalize="fixed")
    
    ious_f_img, cds_f_img, incorrect_f_img, failed_intersect_f_img = evaluate_model_mm(config, model, processor, eval_data_fusion,
                                                                    0, collate_fn,
                                                                    batch_size=500, normalize="fixed")

    eval_data_deepcad.mode = 'pc'
    eval_data_fusion.mode = 'pc'
    ious, cds,  incorrect, failed_intersect = evaluate_model_mm(config, model, processor, eval_data_deepcad, 0,
                                                            collate_fn,
                                                            batch_size=500, normalize="fixed")
    
    ious_f, cds_f, incorrect_f, failed_intersect_f = evaluate_model_mm(config, model, processor, eval_data_fusion,
                                                                    0, collate_fn,
                                                                    batch_size=500, normalize="fixed")

    eval_data_deepcad.mode = 'pc'
    eval_data_fusion.mode = 'pc'
    iou_norm, cd_norm, incorrect_norm, failed_intersect_norm = evaluate_model_mm(
        config, model, processor, eval_data_deepcad, 0, collate_fn, batch_size=500, normalize="elastic"
    )

    iou_norm_f, cd_norm_f, incorrect_norm_f, failed_intersect_norm_f = evaluate_model_mm(
        config, model, processor, eval_data_fusion, 0, collate_fn, batch_size=500, normalize="elastic"
    )

    eval_data_deepcad.mode = 'img'
    eval_data_fusion.mode = 'img'
    iou_norm_img, cd_norm_img, incorrect_norm_img, failed_intersect_norm_img = evaluate_model_mm(
        config, model, processor, eval_data_deepcad, 0, collate_fn, batch_size=500, normalize="elastic"
    )

    iou_norm_f_img, cd_norm_f_img, incorrect_norm_f_img, failed_intersect_norm_f_img = evaluate_model_mm(
        config, model, processor, eval_data_fusion, 0, collate_fn, batch_size=500, normalize="elastic"
    )

    
    experiment.log_metrics({
        # Point‐cloud DeepCAD
        "eval/pc/DeepCAD test/IoU mean":   np.mean(ious),
        "eval/pc/DeepCAD test/CD mean":    np.mean(cds),
        "eval/pc/DeepCAD test/IoU median": np.median(ious),
        "eval/pc/DeepCAD test/CD median":  np.median(cds),

        # Point‐cloud Fusion360
        "eval/pc/Fusion360 test/IoU mean":   np.mean(ious_f),
        "eval/pc/Fusion360 test/CD mean":    np.mean(cds_f),
        "eval/pc/Fusion360 test/IoU median": np.median(ious_f),
        "eval/pc/Fusion360 test/CD median":  np.median(cds_f),

        # Image DeepCAD
        "eval/img/DeepCAD test/IoU mean":   np.mean(ious_img),
        "eval/img/DeepCAD test/CD mean":    np.mean(cds_img),
        "eval/img/DeepCAD test/IoU median": np.median(ious_img),
        "eval/img/DeepCAD test/CD median":  np.median(cds_img),

        # Image Fusion360
        "eval/img/Fusion360 test/IoU mean":   np.mean(ious_f_img),
        "eval/img/Fusion360 test/CD mean":    np.mean(cds_f_img),
        "eval/img/Fusion360 test/IoU median": np.median(ious_f_img),
        "eval/img/Fusion360 test/CD median":  np.median(cds_f_img),
    })

    experiment.log_metrics({
        # Point‐cloud DeepCAD
        "eval/pc/DeepCAD test/IoU mean adaptable norm": np.mean(iou_norm),
        "eval/pc/DeepCAD test/CD mean adaptable norm":      np.mean(cd_norm),
        "eval/pc/DeepCAD test/IoU median adaptable norm": np.median(iou_norm),
        "eval/pc/DeepCAD test/CD median adaptable norm":    np.median(cd_norm),

        # Point‐cloud Fusion360
        "eval/pc/Fusion360 test/IoU mean adaptable norm": np.mean(iou_norm_f),
        "eval/pc/Fusion360 test/CD mean adaptable norm":    np.mean(cd_norm_f),
        "eval/pc/Fusion360 test/IoU median adaptable norm": np.median(iou_norm_f),
        "eval/pc/Fusion360 test/CD median adaptable norm":  np.median(cd_norm_f),

        # Image DeepCAD
        "eval/img/DeepCAD test/IoU mean adaptable norm": np.mean(iou_norm_img),
        "eval/img/DeepCAD test/CD mean adaptable norm":     np.mean(cd_norm_img),
        "eval/img/DeepCAD test/IoU median adaptable norm": np.median(iou_norm_img),
        "eval/img/DeepCAD test/CD median adaptable norm":   np.median(cd_norm_img),

        # Image Fusion360
        "eval/img/Fusion360 test/IoU mean adaptable norm": np.mean(iou_norm_f_img),
        "eval/img/Fusion360 test/CD mean adaptable norm":   np.mean(cd_norm_f_img),
        "eval/img/Fusion360 test/IoU median adaptable norm": np.median(iou_norm_f_img),
        "eval/img/Fusion360 test/CD median adaptable norm": np.median(cd_norm_f_img),
    })

    """
    experiment.log_metrics({
        # Point‐cloud DeepCAD
        "eval/pc/DeepCAD test/AUC mean":            np.mean(aucs),
        "eval/pc/DeepCAD test/Mean Cos Sim  mean":  np.mean(cos_sims),
        "eval/pc/DeepCAD test/AUC median":          np.median(aucs),
        "eval/pc/DeepCAD test/Mean Cos Sim median": np.median(cos_sims),

        # Point‐cloud Fusion360
        "eval/pc/Fusion360 test/AUC mean":            np.mean(aucs_f),
        "eval/pc/Fusion360 test/Mean Cos Sim  mean":  np.mean(cos_sims_f),
        "eval/pc/Fusion360 test/AUC median":          np.median(aucs_f),
        "eval/pc/Fusion360 test/Mean Cos Sim median": np.median(cos_sims_f),

        # Image DeepCAD
        "eval/img/DeepCAD test/AUC mean":            np.mean(aucs_img),
        "eval/img/DeepCAD test/Mean Cos Sim  mean":  np.mean(cos_sims_img),
        "eval/img/DeepCAD test/AUC median":          np.median(aucs_img),
        "eval/img/DeepCAD test/Mean Cos Sim median": np.median(cos_sims_img),

        # Image Fusion360
        "eval/img/Fusion360 test/AUC mean":            np.mean(aucs_f_img),
        "eval/img/Fusion360 test/Mean Cos Sim  mean":  np.mean(cos_sims_f_img),
        "eval/img/Fusion360 test/AUC median":          np.median(aucs_f_img),
        "eval/img/Fusion360 test/Mean Cos Sim median": np.median(cos_sims_f_img),
    })"""

    return

    


if __name__=="__main__":
    main()