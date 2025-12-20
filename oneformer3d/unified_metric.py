import torch
import numpy as np
from scipy import stats
from mmengine.logging import MMLogger

from mmdet3d.evaluation import InstanceSegMetric
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval


@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # These are specific to ForAINetV2 evaluation script
        # We only have wood and leaf (no ground, no unclassified)
        NUM_CLASSES_BINARY = 3  # 0:unclassified, 1:non-tree, 2:tree
        NUM_CLASSES_SEM = 2  # 0:wood, 1:leaf (simple, no unclassified)
        # INS_CLASS_IDS refers to binary array indices: 0=unclassified, 1=non-tree, 2=tree
        # Both wood and leaf map to binary "tree" (2)
        INS_CLASS_IDS = [2]  # Binary "tree" class (both wood and leaf map here)

        # Global accumulators
        true_positive_classes_global = np.zeros(NUM_CLASSES_SEM)
        positive_classes_global = np.zeros(NUM_CLASSES_SEM)
        gt_classes_global = np.zeros(NUM_CLASSES_SEM)
        true_positive_classes_bi_global = np.zeros(NUM_CLASSES_BINARY)
        positive_classes_bi_global = np.zeros(NUM_CLASSES_BINARY)
        gt_classes_bi_global = np.zeros(NUM_CLASSES_BINARY)
        total_gt_ins_global = np.zeros(NUM_CLASSES_BINARY)
        tpsins_global = [[] for _ in range(NUM_CLASSES_BINARY)]
        fpsins_global = [[] for _ in range(NUM_CLASSES_BINARY)]
        IoU_Tp_global = np.zeros(NUM_CLASSES_BINARY)
        all_mean_cov_global = [[] for _ in range(NUM_CLASSES_BINARY)]
        all_mean_weighted_cov_global = [[] for _ in range(NUM_CLASSES_BINARY)]

        for eval_ann, single_pred_results in results:
            # Get GT and Pred labels
            # IMPORTANT: GT might still be 1-indexed from .pkl files, convert to 0-indexed
            sem_gt_i = eval_ann['pts_semantic_mask'].copy()
            # Convert GT from 1-indexed (1=wood, 2=leaf) to 0-indexed (0=wood, 1=leaf) if needed
            if sem_gt_i.min() >= 1:  # GT is 1-indexed
                sem_gt_i = sem_gt_i - 1  # Convert: 1->0, 2->1
            
            # Get predictions - USE INDEX 0 (semantic), NOT INDEX 1 (panoptic)!
            # pts_semantic_mask[0] = pure semantic predictions
            # pts_semantic_mask[1] = panoptic predictions (encoded with instance IDs)
            sem_pre_i = single_pred_results['pts_semantic_mask'][0].copy()
            # Clamp predictions to valid semantic class range [0, NUM_CLASSES_SEM-1] = [0, 1]
            sem_pre_i = np.clip(sem_pre_i, 0, NUM_CLASSES_SEM - 1)
            
            ins_gt_i = eval_ann['pts_instance_mask']
            ins_pre_i = single_pred_results['pts_instance_mask'][1]
            
            # Debug: Check label ranges (only on first sample)
            if len(gt_classes_global) == 0 or np.sum(gt_classes_global) == 0:
                gt_unique = np.unique(sem_gt_i)
                pred_unique = np.unique(sem_pre_i)
                logger.info(f"Debug: GT labels range: {gt_unique}, Pred labels range: {pred_unique}")
                logger.info(f"Debug: GT labels shape: {sem_gt_i.shape}, Pred labels shape: {sem_pre_i.shape}")
                logger.info(f"Debug: NUM_CLASSES_SEM: {NUM_CLASSES_SEM}, NUM_CLASSES_BINARY: {NUM_CLASSES_BINARY}")

            # Ensure all arrays have the same size to avoid index out of bounds
            min_size = min(sem_gt_i.shape[0], sem_pre_i.shape[0], 
                          ins_gt_i.shape[0], ins_pre_i.shape[0])
            sem_gt_i = sem_gt_i[:min_size]
            sem_pre_i = sem_pre_i[:min_size]
            ins_gt_i = ins_gt_i[:min_size]
            ins_pre_i = ins_pre_i[:min_size]

            # Semantic Segmentation Evaluation
            # Labels are already 0-indexed and clamped above
            for j in range(min_size):
                gt_l = int(sem_gt_i[j])
                pred_l = int(sem_pre_i[j])
                # Double-check bounds (should already be clamped, but safety check)
                gt_l = max(0, min(gt_l, NUM_CLASSES_SEM - 1))
                pred_l = max(0, min(pred_l, NUM_CLASSES_SEM - 1))
                # Count all labels (0 and 1 are valid classes)
                gt_classes_global[gt_l] += 1
                positive_classes_global[pred_l] += 1
                if gt_l == pred_l:
                    true_positive_classes_global[gt_l] += 1

            # Binary Semantic and Instance Evaluation
            # Map semantic labels to binary: 2 for thing (wood, leaf)
            # Labels are 0-indexed: 0=wood, 1=leaf, both map to binary 'tree' (2)
            # Initialize binary arrays with 0 (unclassified)
            sem_gt_bi = np.zeros_like(sem_gt_i, dtype=np.int64)
            sem_pre_bi = np.zeros_like(sem_pre_i, dtype=np.int64)
            # All thing classes (wood=0, leaf=1) map to binary 'tree' (2)
            # Only map valid classes (0, 1) to tree (2)
            for tc in self.thing_class_inds:
                if 0 <= tc < NUM_CLASSES_SEM:  # Safety check
                    sem_gt_bi[sem_gt_i == tc] = 2
                    sem_pre_bi[sem_pre_i == tc] = 2

            # Use minimum size to avoid index out of bounds
            min_size_bi = min(sem_gt_bi.shape[0], sem_pre_bi.shape[0])
            for j in range(min_size_bi):
                gt_l = int(sem_gt_bi[j])
                pred_l = int(sem_pre_bi[j])
                # CRITICAL: Clamp to binary array range [0, NUM_CLASSES_BINARY-1]
                # Binary array: 0=unclassified, 1=non-tree, 2=tree
                gt_l = max(0, min(gt_l, NUM_CLASSES_BINARY - 1))
                pred_l = max(0, min(pred_l, NUM_CLASSES_BINARY - 1))
                gt_classes_bi_global[gt_l] += 1
                positive_classes_bi_global[pred_l] += 1
                true_positive_classes_bi_global[gt_l] += int(gt_l == pred_l)

            # Filter out unclassified points (binary class 0) for instance evaluation
            # Keep only tree points (binary class 2)
            idxc = (sem_gt_bi == 2) | (sem_pre_bi == 2)
            pred_ins, gt_ins = ins_pre_i[idxc], ins_gt_i[idxc]
            pred_sem, gt_sem = sem_pre_bi[idxc], sem_gt_bi[idxc]

            # Get predicted instances
            un = np.unique(pred_ins)
            pts_in_pred = [[] for _ in range(NUM_CLASSES_BINARY)]
            for g in un:
                if g == -1: continue
                tmp = (pred_ins == g)
                sem_seg_i = int(stats.mode(pred_sem[tmp], keepdims=True)[0][0])
                pts_in_pred[sem_seg_i].append(tmp)

            # Get ground truth instances
            un = np.unique(gt_ins)
            pts_in_gt = [[] for _ in range(NUM_CLASSES_BINARY)]
            for g in un:
                if g == 0: continue # In ForAINetV2, instance ID 0 is not a valid instance
                tmp = (gt_ins == g)
                sem_seg_i = int(stats.mode(gt_sem[tmp], keepdims=True)[0][0])
                pts_in_gt[sem_seg_i].append(tmp)

            # Coverage Metrics (MUCov, MWCov)
            for i_sem in INS_CLASS_IDS:
                if not pts_in_gt[i_sem] or not pts_in_pred[i_sem]: continue
                sum_cov, num_gt_point, mean_weighted_cov = 0, 0, 0
                for ins_gt in pts_in_gt[i_sem]:
                    ovmax = 0.
                    num_ins_gt_point = np.sum(ins_gt)
                    num_gt_point += num_ins_gt_point
                    for ins_pred in pts_in_pred[i_sem]:
                        union = (ins_pred | ins_gt)
                        intersect = (ins_pred & ins_gt)
                        iou = float(np.sum(intersect)) / np.sum(union)
                        if iou > ovmax: ovmax = iou
                    sum_cov += ovmax
                    mean_weighted_cov += ovmax * num_ins_gt_point
                if len(pts_in_gt[i_sem]) != 0:
                    all_mean_cov_global[i_sem].append(sum_cov / len(pts_in_gt[i_sem]))
                    all_mean_weighted_cov_global[i_sem].append(mean_weighted_cov / num_gt_point)

            # PQ, SQ, RQ Metrics
            for i_sem in INS_CLASS_IDS:
                tp, fp = [0.] * len(pts_in_pred[i_sem]), [0.] * len(pts_in_pred[i_sem])
                IoU_Tp_per = 0
                if pts_in_gt[i_sem]: total_gt_ins_global[i_sem] += len(pts_in_gt[i_sem])

                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    ovmax = -1.
                    if not pts_in_gt[i_sem]:
                        fp[ip] = 1; continue
                    for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                        union = (ins_pred | ins_gt)
                        intersect = (ins_pred & ins_gt)
                        iou = float(np.sum(intersect)) / np.sum(union)
                        if iou > ovmax: ovmax = iou
                    if ovmax >= 0.5:
                        tp[ip] = 1
                        IoU_Tp_per += ovmax
                    else:
                        fp[ip] = 1
                tpsins_global[i_sem].extend(tp)
                fpsins_global[i_sem].extend(fp)
                IoU_Tp_global[i_sem] += IoU_Tp_per

        # Final Metric Calculation
        metrics = dict()

        # Semantic Segmentation
        iou_list = []
        valid_sem_classes = [i for i, n in enumerate(gt_classes_global) if n > 0] # All classes with points
        for i in range(NUM_CLASSES_SEM):
            iou = true_positive_classes_global[i] / float(gt_classes_global[i] + positive_classes_global[i] - true_positive_classes_global[i] + 1e-8)
            iou_list.append(iou)
        metrics['mIoU'] = np.mean([iou_list[i] for i in valid_sem_classes]) if valid_sem_classes else 0.0

        # Binary Semantic Segmentation
        iou_list_bi = []
        valid_bi_sem_classes = [i for i, n in enumerate(gt_classes_bi_global) if n > 0 and i > 0]
        for i in range(1, NUM_CLASSES_BINARY):
            iou = true_positive_classes_bi_global[i] / float(gt_classes_bi_global[i] + positive_classes_bi_global[i] - true_positive_classes_bi_global[i] + 1e-8)
            iou_list_bi.append(iou)
        metrics['mIoU_binary'] = np.mean([iou_list_bi[i-1] for i in valid_bi_sem_classes]) if valid_bi_sem_classes else 0.0

        # Instance Segmentation
        MUCov = np.zeros(NUM_CLASSES_BINARY)
        MWCov = np.zeros(NUM_CLASSES_BINARY)
        precision = np.zeros(NUM_CLASSES_BINARY)
        recall = np.zeros(NUM_CLASSES_BINARY)
        RQ = np.zeros(NUM_CLASSES_BINARY)
        SQ = np.zeros(NUM_CLASSES_BINARY)
        PQ = np.zeros(NUM_CLASSES_BINARY)
        for i_sem in INS_CLASS_IDS:
            MUCov[i_sem] = np.mean(all_mean_cov_global[i_sem]) if all_mean_cov_global[i_sem] else 0
            MWCov[i_sem] = np.mean(all_mean_weighted_cov_global[i_sem]) if all_mean_weighted_cov_global[i_sem] else 0
            tp = np.sum(tpsins_global[i_sem])
            fp = np.sum(fpsins_global[i_sem])
            rec = tp / (total_gt_ins_global[i_sem] + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            precision[i_sem], recall[i_sem] = prec, rec
            RQ[i_sem] = 2 * prec * rec / (prec + rec + 1e-8)
            SQ[i_sem] = IoU_Tp_global[i_sem] / (tp + 1e-8)
            PQ[i_sem] = SQ[i_sem] * RQ[i_sem]
        valid_ins_classes = [i for i in INS_CLASS_IDS if total_gt_ins_global[i] > 0]
        if not valid_ins_classes: valid_ins_classes = INS_CLASS_IDS # Avoid division by zero if no GT

        metrics['mMWCov'] = np.mean(MWCov[valid_ins_classes])
        metrics['mMUCov'] = np.mean(MUCov[valid_ins_classes])
        metrics['mPrecision'] = np.mean(precision[valid_ins_classes])
        metrics['mRecall'] = np.mean(recall[valid_ins_classes])
        metrics['F1'] = (2 * metrics['mPrecision'] * metrics['mRecall']) / (metrics['mPrecision'] + metrics['mRecall'] + 1e-8)
        metrics['mPQ'] = np.mean(PQ[valid_ins_classes])
        metrics['mSQ'] = np.mean(SQ[valid_ins_classes])
        metrics['mRQ'] = np.mean(RQ[valid_ins_classes])

        log_str = 'Evaluation Results:\n'
        log_str += f"mIoU: {metrics['mIoU']:.4f}, mIoU_binary: {metrics['mIoU_binary']:.4f}\n"
        log_str += f"mPQ: {metrics['mPQ']:.4f}, mSQ: {metrics['mSQ']:.4f}, mRQ: {metrics['mRQ']:.4f}\n"
        log_str += f"mPrecision: {metrics['mPrecision']:.4f}, mRecall: {metrics['mRecall']:.4f}, F1: {metrics['F1']:.4f}\n"
        log_str += f"mMUCov: {metrics['mMUCov']:.4f}, mMWCov: {metrics['mMWCov']:.4f}"
        logger.info(log_str)
        
        return metrics


@METRICS.register_module()
class InstanceSegMetric_(InstanceSegMetric):
    """The only difference with InstanceSegMetric is that following ScanNet
    evaluator we accept instance prediction as a boolean tensor of shape
    (n_points, n_instances) instead of integer tensor of shape (n_points, ).

    For this purpose we only replace instance_seg_eval call.
    """

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta['classes']
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            gt_instance_masks.append(eval_ann['pts_instance_mask'])
            pred_instance_masks.append(
                single_pred_results['pts_instance_mask'])
            pred_instance_labels.append(single_pred_results['instance_labels'])
            pred_instance_scores.append(single_pred_results['instance_scores'])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger)

        return ret_dict
