# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.ops.nms import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.models.task_modules import SamplingResult

from mmdet.registry import MODELS as MMDETMODELS
from mmtrack.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, reduce_mean)

from mmdet.models.dense_heads import YOLOXHead
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances, multi_apply)


import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.models.utils import filter_scores_and_topk
from mmdet.utils import ConfigType, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.model import ModuleList, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

@MMDETMODELS.register_module()
class YOLOX_QDTrackHead(YOLOXHead):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        embed_channels (int): The input channel of embed features.
            Defaults to 256.
    """

    def __init__(
            self,
            embed_channels=256,
            softmax_temp: int = -1,
            loss_track: Optional[dict] = None,
            loss_track_aux: dict = dict(
                type='L2Loss',
                # TODO: Why is not working?
                # sample_ratio=3,
                # margin=0.3,
                # loss_weight=1.0,
                # hard_mining=True,
                _scope_='mmtrack'),
            *args, **kwargs) -> None:
        self.embed_channels = embed_channels
        self.softmax_temp = softmax_temp

        super().__init__(*args, **kwargs)
        
        if loss_track is None:
            loss_track = dict(
                type='MultiPosCrossEntropyLoss', loss_weight=0.25, _scope_='mmtrack')

        self.loss_track = MODELS.build(loss_track)

        if loss_track_aux is not None:
            self.loss_track_aux = MODELS.build(loss_track_aux)
        else:
            self.loss_track_aux = None


    def _init_layers(self) -> None:
        """Initialize heads for all level feature maps."""
        super()._init_layers()
        self.multi_level_embed_convs = nn.ModuleList()
        self.multi_level_conv_embed = nn.ModuleList()

        for _ in self.strides:
            self.multi_level_embed_convs.append(self._build_stacked_convs())
            conv_embed = self._build_embed_predictor()
            self.multi_level_conv_embed.append(conv_embed)
    
    def _build_embed_predictor(self) -> nn.Module:
        """Initialize predictor layers of a single level head."""
        return nn.Conv2d(self.feat_channels, self.embed_channels, 1)
    
    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_embed in self.multi_level_conv_embed:
            conv_embed.bias.data.fill_(bias_init)

    def forward_single(self, x: Tensor, cls_convs: nn.Module,
                       reg_convs: nn.Module, conv_cls: nn.Module,
                       embed_convs: nn.Module,
                       conv_reg: nn.Module,
                       conv_obj: nn.Module,
                       conv_embed: nn.Module
                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        cls_score, bbox_pred, objectness = super().forward_single(
            x, cls_convs, reg_convs, conv_cls,
            conv_reg, conv_obj
        )

        embed_feat = embed_convs(x)
        embed_pred = conv_embed(embed_feat)

        return cls_score, bbox_pred, embed_pred, objectness

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """

        return multi_apply(self.forward_single, x, self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_embed_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj,
                           self.multi_level_conv_embed)
    
    def loss_tracking(self,
                   x: Tuple[Tensor], batch_data_samples: SampleList,
                   ref_x: Tuple[Tensor], ref_batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            ref_x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor. Reference frame.
            ref_batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
                Reference frame.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat_track(*loss_inputs)

        ref_outs = self(ref_x)

        ref_outputs = unpack_gt_instances(ref_batch_data_samples)
        (ref_batch_gt_instances, ref_batch_gt_instances_ignore,
         ref_batch_img_metas) = ref_outputs

        ref_loss_inputs = ref_outs + (ref_batch_gt_instances, ref_batch_img_metas,
                              ref_batch_gt_instances_ignore)
        ref_losses = self.loss_by_feat_track(*ref_loss_inputs)

        # Preparing Key frame features for track loss
        (cls_score, bbox_pred, embed_pred, objectness) = outs
        (
            num_imgs, batch_gt_instances_ignore,
            flatten_cls_preds, flatten_bbox_preds,
            flatten_embed_preds, flatten_objectness,
            flatten_priors, flatten_bboxes,
        ) = self._flatten_features(
            cls_score,
            bbox_pred,
            embed_pred,
            objectness,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore
        )
        sampling_result, _ = self._get_sampling_and_assign_results(
            flatten_priors,
            flatten_cls_preds,
            flatten_bbox_preds,
            flatten_embed_preds,
            flatten_objectness,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore
        )
        # Preparing Ref frame features for track loss
        (ref_cls_score, ref_bbox_pred, ref_embed_pred, ref_objectness) = ref_outs
        (
            ref_num_imgs, ref_batch_gt_instances_ignore,
            ref_flatten_cls_preds, ref_flatten_bbox_preds,
            ref_flatten_embed_preds, ref_flatten_objectness,
            ref_flatten_priors, ref_flatten_bboxes,
        ) = self._flatten_features(
            ref_cls_score,
            ref_bbox_pred,
            ref_embed_pred,
            ref_objectness,
            ref_batch_gt_instances,
            ref_batch_img_metas,
            ref_batch_gt_instances_ignore
        )
        ref_sampling_result, _ = self._get_sampling_and_assign_results(
            ref_flatten_priors,
            ref_flatten_cls_preds,
            ref_flatten_bbox_preds,
            ref_flatten_embed_preds,
            ref_flatten_objectness,
            ref_batch_gt_instances,
            ref_batch_img_metas,
            ref_batch_gt_instances_ignore
        )

        loss_track = self.loss_embed_head(
            flatten_embed_preds, sampling_result,
            ref_flatten_embed_preds, ref_sampling_result
        )


        return losses
    
    def loss_embed_head(self, key_roi_feats: Tensor, ref_roi_feats: Tensor,
             key_sampling_results: List[SamplingResult],
             ref_sampling_results: List[SamplingResult],
             gt_match_indices_list: List[Tensor]) -> dict:
        """Calculate the track loss and the auxiliary track loss.

        Args:
            key_roi_feats (Tensor): Embeds of positive bboxes in sampling
                results of key image.
            ref_roi_feats (Tensor): Embeds of all bboxes in sampling results
                of the reference image.
            key_sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            ref_sampling_results (List[obj:SamplingResults]): Assign results of
                all reference images in a batch after sampling.
            gt_match_indices_list (list(Tensor)): Mapping from gt_instances_id
                to ref_gt_instances_id of the same tracklet in a pair of
                images.

        Returns:
            Dict [str: Tensor]: Calculation results.
            Containing the following list of Tensors:

                - loss_track (Tensor): Results of loss_track function.
                - loss_track_aux (Tensor): Results of loss_track_aux function.
        """
        key_track_feats = self(key_roi_feats)
        ref_track_feats = self(ref_roi_feats)

        losses = self.loss_by_feat(key_track_feats, ref_track_feats,
                                   key_sampling_results, ref_sampling_results,
                                   gt_match_indices_list)
        return losses

    def loss_by_feat_emned_head(self, key_track_feats: Tensor, ref_track_feats: Tensor,
                     key_sampling_results: List[SamplingResult],
                     ref_sampling_results: List[SamplingResult],
                     gt_match_indices_list: List[Tensor]) -> dict:
        """Calculate the track loss and the auxiliary track loss.

        Args:
            key_track_feats (Tensor): Embeds of positive bboxes in sampling
                results of key image.
            ref_track_feats (Tensor): Embeds of all bboxes in sampling results
                of the reference image.
            key_sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            ref_sampling_results (List[obj:SamplingResults]): Assign results of
                all reference images in a batch after sampling.
            gt_match_indices_list (list(Tensor)): Mapping from gt_instances_id
                to ref_gt_instances_id of the same tracklet in a pair of
                images.

        Returns:
            Dict [str: Tensor]: Calculation results.
            Containing the following list of Tensors:

                - loss_track (Tensor): Results of loss_track function.
                - loss_track_aux (Tensor): Results of loss_track_aux function.
        """
        dists, cos_dists = self.match(key_track_feats, ref_track_feats,
                                      key_sampling_results,
                                      ref_sampling_results)
        targets, weights = self.get_targets(gt_match_indices_list,
                                            key_sampling_results,
                                            ref_sampling_results)
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum())
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / len(dists)

        return losses
    
    def _flatten_features(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            embed_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> tuple:
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_embed_preds = [
            embed_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.embed_channels)
            for embed_pred in embed_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_embed_preds = torch.cat(flatten_embed_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        return (
            num_imgs,
            batch_gt_instances_ignore,
            flatten_cls_preds,
            flatten_bbox_preds,
            flatten_embed_preds,
            flatten_objectness,
            flatten_priors,
            flatten_bboxes,
        )

    def loss_by_feat_track(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            embed_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            embed_preds (list[Tensor]): Embdings predictions for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * embed size, H, W).
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_embed_preds = [
            embed_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.embed_channels)
            for embed_pred in embed_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_embed_preds = torch.cat(flatten_embed_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_targets_single,
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_cls_preds.detach(),
             flatten_bboxes.detach(), flatten_embed_preds.detach(),
             flatten_objectness.detach(), batch_gt_instances, batch_img_metas,
             batch_gt_instances_ignore)

        # The experimental results show that 'reduce_mean' can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        if num_pos > 0:
            loss_cls = self.loss_cls(
                flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                cls_targets) / num_total_samples
            loss_bbox = self.loss_bbox(
                flatten_bboxes.view(-1, 4)[pos_masks],
                bbox_targets) / num_total_samples
        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            if num_pos > 0:
                loss_l1 = self.loss_l1(
                    flatten_bbox_preds.view(-1, 4)[pos_masks],
                    l1_targets) / num_total_samples
            else:
                # Avoid cls and reg branch not participating in the gradient
                # propagation when there is no ground-truth in the images.
                # For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_l1 = flatten_bbox_preds.sum() * 0
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        embed_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            embed_preds (list[Tensor]): Embdings predictions for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * embed size, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_embed_preds = [
            embed_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.embed_channels)
            for embed_pred in embed_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_embed_preds = torch.cat(flatten_embed_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], 1)
            valid_mask = flatten_objectness[
                img_id] * max_scores >= cfg.score_thr
            results = InstanceData(
                bboxes=flatten_bboxes[img_id][valid_mask],
                scores=max_scores[valid_mask] *
                flatten_objectness[img_id][valid_mask],
                labels=labels[valid_mask],
                embeds=flatten_embed_preds[valid_mask])

            result_list.append(
                self._bbox_post_process(
                    results=results,
                    cfg=cfg,
                    rescale=rescale,
                    with_nms=with_nms,
                    img_meta=img_meta))

        return result_list

    @torch.no_grad()
    def _get_sampling_and_assign_results(
            self,
            priors: Tensor,
            cls_preds: Tensor,
            decoded_bboxes: Tensor,
            objectness: Tensor,
            gt_instances: InstanceData,
            img_meta: dict,
            gt_instances_ignore: Optional[InstanceData] = None
    ): # -> What does it return?!
        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(
            bboxes=decoded_bboxes, scores=scores.sqrt_(), priors=offset_priors)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        return sampling_result, assign_result
    
    @torch.no_grad()
    def _get_targets_single(
            self,
            priors: Tensor,
            cls_preds: Tensor,
            decoded_bboxes: Tensor,
            objectness: Tensor,
            gt_instances: InstanceData,
            img_meta: dict,
            gt_instances_ignore: Optional[InstanceData] = None) -> tuple:
        """Compute classification, regression, and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            tuple:
                foreground_mask (list[Tensor]): Binary mask of foreground
                targets.
                cls_target (list[Tensor]): Classification targets of an image.
                obj_target (list[Tensor]): Objectness targets of an image.
                bbox_target (list[Tensor]): BBox targets of an image.
                l1_target (int): BBox L1 targets of an image.
                num_pos_per_img (int): Number of positive samples in an image.
        """
        sampling_result, assign_result = self._get_sampling_and_assign_results(
            priors,
            cls_preds,
            decoded_bboxes,
            objectness,
            gt_instances,
            img_meta,
            gt_instances_ignore
        )
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)
