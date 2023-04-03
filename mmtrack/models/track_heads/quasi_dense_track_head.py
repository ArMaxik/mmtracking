# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmdet.structures.bbox import bbox2roi
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.samplers.sampling_result import SamplingResult
from torch import Tensor
import torch

from mmtrack.registry import MODELS
from mmtrack.utils import InstanceList, SampleList
from .roi_track_head import RoITrackHead


@MODELS.register_module()
class QuasiDenseTrackHead(RoITrackHead):
    """The quasi-dense track head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_roi_feats(self, feats: List[Tensor],
                          bboxes: List[Tensor]) -> Tensor:
        """Extract roi features.

        Args:
            feats (list[Tensor]): list of multi-level image features.
            bboxes (list[Tensor]): list of bboxes in sampling result.

        Returns:
            Tensor: The extracted roi features.
        """
        rois = bbox2roi(bboxes)
        bbox_feats = self.roi_extractor(feats[:self.roi_extractor.num_inputs],
                                        rois)
        return bbox_feats

    def loss(self, key_feats: List[Tensor], ref_feats: List[Tensor],
             rpn_results_list: InstanceList,
             ref_rpn_results_list: InstanceList, data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            key_feats (list[Tensor]): list of multi-level image features.
            ref_feats (list[Tensor]): list of multi-level ref_img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals of key img.
            ref_rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals of ref img.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        assert self.with_track
        num_imgs = len(data_samples)
        batch_gt_instances = []
        ref_batch_gt_instances = []
        batch_gt_instances_ignore = []
        gt_match_indices_list = []
        for data_sample in data_samples:
            batch_gt_instances.append(data_sample.gt_instances)
            ref_batch_gt_instances.append(data_sample.ref_gt_instances)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
            # get gt_match_indices
            ins_ids = data_sample.gt_instances.instances_id.tolist()
            ref_ins_ids = data_sample.ref_gt_instances.instances_id.tolist()
            match_indices = Tensor([
                ref_ins_ids.index(i) if (i in ref_ins_ids and i > 0) else -1
                for i in ins_ids
            ]).to(key_feats[0].device)
            gt_match_indices_list.append(match_indices)

        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            ref_rpn_results = ref_rpn_results_list[i]
            # rename ref_rpn_results.bboxes to ref_rpn_results.priors
            ref_rpn_results.priors = ref_rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in key_feats])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.bbox_assigner.assign(
                ref_rpn_results, ref_batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            ref_sampling_result = self.bbox_sampler.sample(
                ref_assign_result,
                ref_rpn_results,
                ref_batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_feats])
            ref_sampling_results.append(ref_sampling_result)

        key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_roi_feats = self.extract_roi_feats(key_feats, key_bboxes)
        ref_bboxes = [res.bboxes for res in ref_sampling_results]
        ref_roi_feats = self.extract_roi_feats(ref_feats, ref_bboxes)

        loss_track = self.embed_head.loss(key_roi_feats, ref_roi_feats,
                                          key_sampling_results,
                                          ref_sampling_results,
                                          gt_match_indices_list)

        return loss_track

    def predict(self, feats: List[Tensor],
                rescaled_bboxes: List[Tensor]) -> Tensor:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            feats (list[Tensor]): Multi level feature maps of `img`.
            rescaled_bboxes (list[Tensor]): list of rescaled bboxes in sampling
                result.

        Returns:
            Tensor: The extracted track features.
        """
        bbox_feats = self.extract_roi_feats(feats, rescaled_bboxes)
        track_feats = self.embed_head.predict(bbox_feats)
        return track_feats


@MODELS.register_module()
class QuasiDenseTrackHeadOneStage(QuasiDenseTrackHead):
    """The quasi-dense track head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def extract_roi_feats(self, feats: List[Tensor],
                          sampling_results: List[SamplingResult],
                          assign_result: List[AssignResult],
                          rpn_results_list: InstanceList,) -> Tensor:
        """Extract roi features.

        Args:
            feats (list[Tensor]): list of multi-level image features.
            bboxes (list[Tensor]): list of bboxes in sampling result.

        Returns:
            Tensor: The extracted roi features.
        """
        # pred_bboxes_inds = [res.pos_inds[(1-res.pos_is_gt).bool()] - res.num_gts for res in sampling_results]
        # feat_inds = [res.valid_mask[pb_ind] for res, pb_ind in zip(rpn_results_list, pred_bboxes_inds)]
        feat_inds = [res.valid_mask[res.pos_inds[res.num_gts:]] for res in sampling_results]
        flat_feats = [torch.cat([f[i].view(f[i].shape[0],-1) for f in feats], dim=1) for i in range(len(assign_result))]
        pred_feats = [f.T[f_ind.long()] for f, f_ind in zip(flat_feats,feat_inds)]
        gt_bboxes = [res.pos_bboxes[:res.num_gts] for res in sampling_results]
        
        roi_features = []
        for pred_f, gt_bbox in zip(pred_feats, gt_bboxes):
            rois = bbox2roi([gt_bbox])
            bbox_feats = self.roi_extractor(feats[:self.roi_extractor.num_inputs],
                                            rois)
            roi_features.append(torch.cat([bbox_feats, torch.unsqueeze(torch.unsqueeze(pred_f, -1), -1)]))
        return torch.cat(roi_features)
    
    def extract_ref_roi_feats(self, feats: List[Tensor],
                          sampling_results: List[SamplingResult],
                          assign_result: List[AssignResult],
                          rpn_results_list: InstanceList,) -> Tensor:
        """Extract roi features.

        Args:
            feats (list[Tensor]): list of multi-level image features.
            bboxes (list[Tensor]): list of bboxes in sampling result.

        Returns:
            Tensor: The extracted roi features.
        """
        # real_gt_num = [(s.pos_is_gt == 1).sum()for s in sampling_results] # WHY ARE THEY DIFFER?!!!
        feat_inds = [res.valid_mask[torch.cat([res.pos_inds[res.num_gts:],res.neg_inds])] for res in sampling_results]
        flat_feats = [torch.cat([f[i].view(f[i].shape[0],-1) for f in feats], dim=1) for i in range(len(assign_result))]
        pred_feats = [f.T[f_ind.long()] for f, f_ind in zip(flat_feats,feat_inds)]
        gt_bboxes = [res.pos_bboxes[:res.num_gts] for res in sampling_results]
        
        roi_features = []
        for pred_f, gt_bbox in zip(pred_feats, gt_bboxes):
            rois = bbox2roi([gt_bbox])
            bbox_feats = self.roi_extractor(feats[:self.roi_extractor.num_inputs],
                                            rois)
            roi_features.append(torch.cat([bbox_feats, torch.unsqueeze(torch.unsqueeze(pred_f, -1), -1)]))
        
        for roi_feature, sampling_result in zip(roi_features, sampling_results):
            assert roi_feature.shape[0] == sampling_result.bboxes.shape[0]
        return torch.cat(roi_features)

    def loss(self, key_feats: List[Tensor], ref_feats: List[Tensor],
             rpn_results_list: InstanceList,
             ref_rpn_results_list: InstanceList, data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            key_feats (list[Tensor]): list of multi-level image features.
            ref_feats (list[Tensor]): list of multi-level ref_img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals of key img.
            ref_rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals of ref img.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        assert self.with_track
        num_imgs = len(data_samples)
        batch_gt_instances = []
        ref_batch_gt_instances = []
        batch_gt_instances_ignore = []
        gt_match_indices_list = []
        for data_sample in data_samples:
            batch_gt_instances.append(data_sample.gt_instances)
            ref_batch_gt_instances.append(data_sample.ref_gt_instances)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
            # get gt_match_indices
            ins_ids = data_sample.gt_instances.instances_id.tolist()
            ref_ins_ids = data_sample.ref_gt_instances.instances_id.tolist()
            match_indices = Tensor([
                ref_ins_ids.index(i) if (i in ref_ins_ids and i > 0) else -1
                for i in ins_ids
            ]).to(key_feats[0].device)
            gt_match_indices_list.append(match_indices)

        key_assign_results, ref_assign_results = [], []
        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            ref_rpn_results = ref_rpn_results_list[i]
            # rename ref_rpn_results.bboxes to ref_rpn_results.priors
            ref_rpn_results.priors = ref_rpn_results.pop('bboxes')
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in key_feats])
            key_sampling_results.append(sampling_result)
            key_assign_results.append(assign_result)

            ref_assign_result = self.bbox_assigner.assign(
                ref_rpn_results, ref_batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            ref_sampling_result = self.bbox_sampler.sample(
                ref_assign_result,
                ref_rpn_results,
                ref_batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_feats])
            ref_sampling_results.append(ref_sampling_result)
            ref_assign_results.append(ref_assign_result)

        # key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_roi_feats = self.extract_roi_feats(
            key_feats,
            key_sampling_results,
            key_assign_results,
            rpn_results_list
        )
        # ref_bboxes = [res.bboxes for res in ref_sampling_results]
        # ref_roi_feats = self.extract_roi_feats(ref_feats, ref_bboxes)
        ref_roi_feats = self.extract_ref_roi_feats(
            ref_feats,
            ref_sampling_results,
            ref_assign_results,
            ref_rpn_results_list
        )

        loss_track = self.embed_head.loss(key_roi_feats, ref_roi_feats,
                                          key_sampling_results,
                                          ref_sampling_results,
                                          gt_match_indices_list)

        return loss_track

    def predict(self, feats: List[Tensor],
                valid_mask: List[Tensor]) -> Tensor:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            feats (list[Tensor]): Multi level feature maps of `img`.
            rescaled_bboxes (list[Tensor]): list of rescaled bboxes in sampling
                result.

        Returns:
            Tensor: The extracted track features.
        """
        flat_feats = [torch.cat([f[i].view(f[i].shape[0],-1) for f in feats], dim=1) for i in range(feats[0].shape[0])]
        pred_feats = [f.T[f_ind.long()] for f, f_ind in zip(flat_feats, [valid_mask])]
        pred_feats= torch.unsqueeze(torch.unsqueeze(pred_feats[0], -1), -1)
        track_feats = self.embed_head.predict(pred_feats)
        return track_feats
