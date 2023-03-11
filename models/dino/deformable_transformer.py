# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='no',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,

                 use_detached_boxes_dec_out=False,
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        """
        d_model: 编码器里面mlp（前馈神经网络  2个linear层）的hidden dim 512
        nhead: 多头注意力头数 8
        num_encoder_layers: encoder的层数 6
        num_decoder_layers: decoder的层数 6
        dim_feedforward: 前馈神经网络的维度 2048
        dropout: 0.1
        activation: 激活函数类型 relu
        normalize_before: 是否使用前置LN
        return_intermediate_dec: 是否返回decoder中间层结果  False
        """
        # 初始化一个小encoder
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points,
                                                              add_channel_attention=add_channel_attention,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # 创建整个Encoder层  6个encoder层堆叠
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,
            encoder_norm, d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type
        )

        # choose decoder layer type
        # 初始化一个小decoder
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type,
                                                              key_aware_type=key_aware_type,
                                                              decoder_sa_type=decoder_sa_type,
                                                              module_seq=module_seq)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        # 创建整个Decoder层  6个decoder层堆叠
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          modulate_hw_attn=modulate_hw_attn,
                                          num_feature_levels=num_feature_levels,
                                          deformable_decoder=deformable_decoder,
                                          decoder_query_perturber=decoder_query_perturber,
                                          dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                          dec_layer_share=dec_layer_share,
                                          use_detached_boxes_dec_out=use_detached_boxes_dec_out
                                          )

        self.d_model = d_model  # 编码器里面mlp的hidden dim 512
        self.nhead = nhead      # 多头注意力头数 8
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        # scale-level position embedding  [4, 256]
        # 因为deformable detr用到了多尺度特征  经过backbone会生成4个不同尺度的特征图  但是如果还是使用原先的sine position embedding
        # detr是针对h和w进行编码的 不同位置的特征点会对应不同的编码值 但是deformable detr不同的特征图的不同位置就有可能会产生相同的位置编码，就无法区分了
        # 为了解决这个问题，这里引入level_embed这个遍历  不同层的特征图会有不同的level_embed 再让原先的每层位置编码+每层的level_embed
        # 这样就很好的区分不同层的位置编码了  而且这个level_embed是可学习的
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == 'standard':
            # anchor selection at the output of encoder
            # 对Encoder输出memory进行处理：全连接层 + 层归一化
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:

                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[
                           0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[
                           0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        # 参考anchor detr, 类似于Anchor的作用
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)  # 在0-1之间随机生成
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])  # 反归一化
            self.refpoint_embed.weight.data[:, :2].requires_grad = False  # 不进行梯度计算

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        """
        Input:num_dn/ d_model =? hidden_dim
            - srcs: List of multi features [batch_size, channel, height, width]
            - masks: List of multi masks [batch_size, height, width]
            - refpoint_embed: [batch_size, num_queries, 4]. None in infer
            - pos_embeds: List of multi pos embeds [batch_size, channel, height, width]
            - tgt: [batch_size, num_queries, d_model]. None in infer
        """
        """
        Input:num_dn/ d_model =? hidden_dim
            - srcs: 一个由不同尺度特征图Tensor构成的List [batch_size, channel, height, width]
            - masks: 一个由不同尺度特征图对应遮罩的List [batch_size, height, width]
            - refpoint_embed: 新增去噪任务与推理任务的归一化坐标 [batch_size, num_queries, 4]. 仅在训练中存在
            - pos_embeds: List of multi pos embeds [batch_size, channel, height, width]
            - tgt: 新增去噪任务与推理任务的标签Embedding [batch_size, num_queries, d_model]. 仅在训练中存在
            - attn_mask: 新增去噪任务与推理任务的遮罩, 防止偷看其他数据 [query_num + dn_number * 2, query_num + dn_number * 2] 仅在训练任务中存在
        """
        # prepare input for encoder
        # 准备encoder的输入参数
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape  # batch_size channel h w
            spatial_shape = (h, w)  # 特征图shape
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # [bs,c,h,w] -> [bs,h*w,c]
            mask = mask.flatten(1)  # [bs,h,w] -> [bs, h*w]
            # pos_embed: detr的位置编码 仅仅可以区分h,w的位置 因此对应不同的特征图有相同的h、w位置的话，是无法区分的
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [bs,c,h,w] -> [bs,h*w,c]
            # scale-level position embedding  [bs,hxw,c] + [1,1,c] -> [bs,hxw,c]
            # 每一层所有位置加上相同的level_embed 且 不同层的level_embed不同
            # 所以这里pos_embed + level_embed，这样即使不同层特征有相同的w和h，那么也会产生不同的lvl_pos_embed  这样就可以区分了
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        # K =  H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{h*w}, c
        # list4[bs, H/8 * W/8] [bs, H/16 * W/16] [bs, H/32 * W/32] [bs, H/64 * W/64] -> [bs, K]
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        # [4, h+w]  4个特征图的高和宽
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # 不同尺度特征图对应被flatten的那个维度的起始索引  Tensor[4]  如[0,15100,18900,19850]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 各尺度特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        # 为decoder的输入作准备：得到参考点、query embedding(tgt)和query pos(query_embed)
        # one-stage和two-stage的生成方式不同
        # two-stage: 参考点=Encoder预测的top-k（300个）得分最高的proposal boxes,然后对参考点进行位置嵌入生成query和query pos
        # one-stage: query和query pos就是预设的query_embed,然后将query_embed经过全连接层输出2d参考点（归一化的中心坐标）

        '''处理encoder输出结果'''

        if self.two_stage_type == 'standard':  # 默认使用二阶段,two_stage_type = standard
            if self.two_stage_learn_wh:  # 默认不进入, two_stage_learn_wh = None
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:  # 默认进入
                input_hw = None
            # (bs, \sum{hw}, c)
            # 对memory进行处理得到output_memory: [bs, H/8 * W/8 + ... + H/64 * W/64, 256]
            # 并生成初步output_proposals: [bs, H/8 * W/8 + ... + H/64 * W/64, 4]  其实就是特征图上的一个个的点坐标
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes,
                                                                           input_hw)
            # 对encoder输出进行处理：全连接层 + LayerNorm
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            if self.two_stage_pat_embed > 0:  # 默认不进入
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:  # 默认不进入
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            # hack implementation for two-stage Deformable DETR
            # 多分类：[bs, H/8 * W/8 + ... + H/64 * W/64, 256] -> [bs, H/8 * W/8 + ... + H/64 * W/64, 91]
            # 把每个特征点头还原分类  # 其实个人觉得这里直接进行一个二分类足够了
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            # 回归：预测偏移量 + 参考点坐标   [bs, H/8 * W/8 + ... + H/64 * W/64, 4]
            # 还原所有检测框  # two-stage 必须和 iterative bounding box refinement一起使用 不然bbox_embed=None 报错
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            # 保留前多少个query  # 得到参考点reference_points/先验框
            topk = self.num_queries
            # 直接用第一个类别的预测结果来算top-k，代表二分类
            # 如果不使用iterative bounding box refinement那么所有class_embed共享参数 导致第二阶段对解码输出进行分类时都会偏向于第一个类别
            # 获取前900个检测结果的位置  # topk_proposals: [bs, 900]  top900 index
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  # bs, nq

            # gather boxes
            # 根据前九百个位置获取对应的参考点  # topk_coords_unact: top300个分类得分最高的index对应的预测bbox [bs, 900, 4]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            # 以先验框的形式存在  取消梯度
            refpoint_embed_ = refpoint_embed_undetach.detach()
            # 得到归一化参考点坐标  最终会送到decoder中作为初始的参考点
            init_box_proposal = torch.gather(output_proposals, 1,
                                             topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()

            # gather tgt  # 根据前九百个位置获取query
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:  # 默认进入  # 如果已经初始化了tgt_embed
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            else:  # 默认不进入
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:  # 训练时进入
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_],
                                           dim=1)  # 将正常query与cdn正负情绪query的bbox拼接900+200
                tgt = torch.cat([tgt, tgt_], dim=1)  # 将初始化query与cdn正负情绪query拼接900+200
            else:  # 默认不进入
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':  # 默认不进入,推理时进入
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, 4

            if refpoint_embed is not None:  # 默认进入
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_],
                                           dim=1)  # 将正常query与cdn正负情绪query的bbox拼接900+200
                tgt = torch.cat([tgt, tgt_], dim=1)  # 将正常query与cdn正负情绪query拼接900+200
            else:  # 默认不进入
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:  # 默认不进入
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries,
                                                                             1)  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            # 将topk的query的bbox归一化
            init_box_proposal = refpoint_embed_.sigmoid()

        else:  # 默认不进入
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))

        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model 1, 1100, 256
        # - refpoint_embed(unsigmoid): bs, NQ, d_model 1, 1100, 4
        ######################################################### 

        #########################################################
        # Begin Decoder
        #########################################################
        # tgt: 初始化query embedding [bs, 900 + 200, 256]
        # memory: Encoder输出结果 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        # mask_flatten: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        # lvl_pos_embed_flatten: query pos[bs, 300, 256]
        # reference_points: 由query pos接一个全连接层 再归一化后的参考点中心坐标 [bs, 900 + 200, 2]  two-stage=[bs, 900 + 200, 4]
        # level_start_index: [4, ] 4个特征层flatten后的开始index
        # spatial_shapes: [4, 2] 4个特征层的shape
        # valid_ratios: [bs, 4, 2] padding比例 全1
        # attn_mask: [900 + 200, 900 + 200] query遮罩,防止偷看
        # hs: 6层decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 900 + 200, 256]
        # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6, bs, 900 + 200, 2]
        #                   one-stage=[n_decoder, bs, num_query, 2]  two-stage=[n_decoder, bs, num_query, 4]
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask)
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################     
        if self.two_stage_type == 'standard':  # 默认进入
            if self.two_stage_keep_all_tokens:  # 默认不进入
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals

            else:  # 默认进入
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:  # 默认不进入
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################        

        # hs: 6层decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 300, 256]
        # init_reference_out: 初始化的参考点归一化中心坐标 [bs, 300, 2]
        # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6, bs, 300, 2]
        #                   one-stage=[n_decoder, bs, num_query, 2]  two-stage=[n_decoder, bs, num_query, 4]
        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 two_stage_type='no',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            # 6层DeformableTransformerEncoderLayer
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1)])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        生成参考点   reference points  为什么参考点是中心点？  为什么要归一化？
        spatial_shapes: 多尺度feature map对应的h,w，shape为[num_level,2]
        valid_ratios: 多尺度feature map对应的mask中有效的宽高比，shape为[bs, num_levels, 2]  如全是1
        device: cuda:0
        """
        reference_points_list = []
        # 遍历4个特征图的shape  比如 H_=100  W_=150
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 0.5 -> 99.5 取100个点  0.5 1.5 2.5 ... 99.5
            # 0.5 -> 149.5 取150个点 0.5 1.5 2.5 ... 149.5
            # ref_y: [100, 150]  第一行：150个0.5  第二行：150个1.5 ... 第100行：150个99.5
            # ref_x: [100, 150]  第一行：0.5 1.5...149.5   100行全部相同
            # 对于每一层feature map初始化每个参考点中心横纵坐标，加减0.5是确保每个初始点是在每个pixel的中心，例如[0.5,1.5,2.5, ...]
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # 将横纵坐标进行归一化，处理成0-1之间的数 [h, w] -> [bs, hw]  150个0.5 + 150个1.5 + ... + 150个99.5 -> 除以100 归一化
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # [h, w] -> [bs, hw]  100个: 0.5 1.5 ... 149.5  -> 除以150 归一化
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # 得到每一层feature map对应的reference point，即ref，shape为[B, flatten_W*flatten_H, 2] 每一项都是xy
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # list4: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2] + [bs, H/64*W/64, 2] ->
        # 将所有尺度的feature map对应的reference point在第一维合并，得到[bs, N, 2] [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points: [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2] -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 1, 2]
        # valid_ratios: [1, 4, 2] -> [1, 1, 4, 2]
        # 从[bs, N, 2]扩充尺度到[bs, N, num_level, 2] 复制4份 每个特征点都有4个归一化参考点 -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 4, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # 4个flatten后特征图的归一化参考点坐标
        return reference_points

    def forward(self,
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                ref_token_index: Optional[Tensor] = None,
                ref_token_coord: Optional[Tensor] = None
                ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        """
        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src
        # preparation and reshape
        # 4个flatten后特征图的归一化参考点坐标 每个特征点有4个参考点 xy坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        # 未进入, 无效参数 默认ref_token_index=None
        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # main process
            dropflag = False  # if not dropflag 一起删除 从未被修改,一定进入if语句
            if self.enc_layer_dropout_prob is not None:  # 未进入 默认enc_layer_dropout_prob=None
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:  # 一定进入
                if self.deformable_encoder:  # 一定使用deformable_encoder 默认为True
                    output = layer(src=output, pos=pos, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                   key_padding_mask=key_padding_mask)
                else:  # 应该不会适应普通encoder 一般不会进入
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1),
                                   key_padding_mask=key_padding_mask).transpose(0, 1)

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1']) \
                or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):  # 默认不会进入,默认two_stage_type = standard
                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))

                # gather boxes
                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # bs, nq
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory

            # aux loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:  # 默认不会进入,因为ref_token_index一定为None
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:  # 不会进入,默认norm=None
            output = self.norm(output)

        if ref_token_index is not None:  # 不会进入, 默认ref_token_index=None
            intermediate_output = torch.stack(intermediate_output)  # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:  # 默认进入
            intermediate_output = intermediate_ref = None  # 默认为空,无意义参数
        # 经过6层encoder增强后的新特征  每一层不断学习特征层中每个位置和4个采样点的相关性，最终输出的特征是增强后的特征图
        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        return output, intermediate_output, intermediate_ref  # output = query,其余为无意义参数


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=False,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 use_detached_boxes_dec_out=False
                 ):
        super().__init__()
        if num_layers > 0:
            # 6层DeformableTransformerDecoderLayer
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers  # 6
        self.norm = norm
        self.return_intermediate = return_intermediate  # True  默认是返回所有Decoder层输出 计算所有层损失
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None  # 策略1  iterative bounding box refinement
        self.class_embed = None  # 策略2  two-stage Deformable DETR

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,

                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        """
        tgt: [900 + 200, bs, 256] 预设的query embedding 
        memory: [H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, bs, 256] 来自backbone的多层特征信息
        tgt_mask: [1100, 1100] query遮罩
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [batch, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, bs, 256] 来自backbone的遮罩信息
        pos: [H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, bs, 256] 来自backbone的位置信息
        refpoints_unsigmoid: [1100, 1, 4] query的参考点 替代了query pos
        level_start_index: [4,] 每层的起始位置
        spatial_shapes: [4, 2] 每层的实际大小
        valid_raitos: [bs, 4, 2] 每层图片padding后实际所占的比例
        """
        output = tgt

        intermediate = []  # 中间各层+首尾两层=6层输出的解码结果
        reference_points = refpoints_unsigmoid.sigmoid()  # 中间各层+首尾两层输出的参考点（不断矫正）
        ref_points = [reference_points]  # 起始有一个+中间各层+首尾两层=7个参考点

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            # 预处理参考点
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:  # 默认不进入,decoder_query_perturber=None
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:  # 默认进入,deformable_decoder=True
                # 得到参考点坐标
                # two stage
                if reference_points.shape[-1] == 4:  # 默认是二阶段
                    # (1100, bs, 4, 4) 拷贝四份,每层一份
                    reference_points_input = reference_points[:, :, None] \
                                             * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
                else:  # 默认不进入
                    # one stage模式下参考点是query pos通过一个全连接层线性变化为2维的  中心坐标形式(x,y)
                    assert reference_points.shape[-1] == 2
                    # [bs, 300, 1, 2] * [bs, 1, 4, 2] -> [bs, 300, 4, 2]=[bs, n_query, n_lvl, 2]
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                # 生成位置编码  根据参考点生成位置编码
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2
            else:  # 默认不进入
                query_sine_embed = gen_sineembed_for_position(reference_points)  # nq, bs, 256*2
                reference_points_input = None

            # conditional query
            # 原生query位置编码
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:  # 默认不进入
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:  # 默认不进入
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(
                    -1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(
                    -1)

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:  # 默认不进入
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:  # 默认进入
                output = layer(
                    tgt=output,  # [1100, 1, 256]上一层的输出
                    tgt_query_pos=query_pos,  # 位置编码
                    tgt_query_sine_embed=query_sine_embed,  # 位置编码的Embedding
                    tgt_key_padding_mask=tgt_key_padding_mask,  # None
                    tgt_reference_points=reference_points_input,  # [1100, 1, 4, 4] nq, bs, nlevel, 4 参考点

                    memory=memory,  # [hw, bs, 256]
                    memory_key_padding_mask=memory_key_padding_mask,  # [bs, hw]
                    memory_level_start_index=level_start_index,  # [4,] 层起始位置
                    memory_spatial_shapes=spatial_shapes,  # [4, 2] 层大小
                    memory_pos=pos,  # [hw, bs, 256] backbone特征位置编码

                    self_attn_mask=tgt_mask,  # [1100, 1100] 遮罩
                    cross_attn_mask=memory_mask  # None
                )

            # iter update
            # hack implementation for iterative bounding box refinement
            # 使用iterative bounding box refinement 这里的self.bbox_embed就不是None
            # 如果没有iterative bounding box refinement那么reference_points是不变的
            # 每层参考点都会根据上一层的输出结果进行矫正
            if self.bbox_embed is not None:  # 默认进入
                reference_before_sigmoid = inverse_sigmoid(reference_points)  # 还原参考点
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                # select # ref points
                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:  # 默认不进入
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output)  # nq, bs, 91
                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1]  # new_nq, bs
                        new_reference_points = torch.gather(new_reference_points, 0,
                                                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid

                if self.rm_detach and 'dec' in self.rm_detach:  # 默认不进入
                    reference_points = new_reference_points
                else:  # 默认进入,更新参考点
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:  # 默认不进入
                    ref_points.append(reference_points)
                else:  # 默认进入
                    ref_points.append(new_reference_points)

            # 默认返回6个decoder层输出一起计算损失
            intermediate.append(self.norm(output))

            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:  # 默认不进入
                if nq_now != select_number:
                    output = torch.gather(output, 0,
                                          topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))  # unsigmoid

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()
        # self attention
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                      used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        """
        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        reference_points: 4个flatten后特征图对应的归一化参考点坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        # self attention + add + norm
        # query = flatten后的多尺度特征图 + scale-level pos
        # key = 采样点  每个特征点对应周围的4个可学习的采样点
        # value = flatten后的多尺度特征图

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn   feed forward + add + norm
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        # cross attention
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points,
                                                       used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                   # for tgt
                   tgt: Optional[Tensor],  # nq, bs, d_model
                   tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                   tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                   # for memory
                   memory: Optional[Tensor] = None,  # hw, bs, d_model
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,  # num_levels
                   memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                   memory_pos: Optional[Tensor] = None,  # pos for memory

                   # sa
                   self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                   cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                   ):
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                # self-attention
                # 第一个attention的目的：学习各个物体之间的关系/位置   可以知道图像当中哪些位置会存在物体  物体信息->tgt
                # 所以qk都是query embedding + query pos   v就是query embedding
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                                      tgt_reference_points.transpose(0, 1).contiguous(),
                                      memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                                      memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                   # for tgt
                   tgt: Optional[Tensor],  # nq, bs, d_model
                   tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                   tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                   # for memory
                   memory: Optional[Tensor] = None,  # hw, bs, d_model
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,  # num_levels
                   memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                   memory_pos: Optional[Tensor] = None,  # pos for memory

                   # sa
                   self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                   cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                   ):
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        # cross attention  使用（多尺度）可变形注意力模块替代原生的Transformer交叉注意力
        # 第二个attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中去，更好地表征了图像中的各个物体
        # 所以q=query embedding + query pos, k = query pos通过一个全连接层->2维, v=上一层输出的output
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                               memory_key_padding_mask).transpose(0, 1)
        # add + norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # [bs, 300, 256]  self-attention输出特征 + cross-attention输出特征
        # 最终的特征：知道图像中物体与物体之间的位置关系 + encoder增强后的图像特征 + 图像与物体之间的关系
        return tgt

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):
        """
        tgt: 预设的query embedding [bs, 300, 256]
        query_pos: 预设的query pos [bs, 300, 256]
        reference_points: query pos通过一个全连接层->2维  [bs, 300, 4, 2] = [bs, num_query, n_layer, 2]
                          iterative bounding box refinement时 = [bs, num_query, n_layer, 4]
        src: 第一层是encoder的输出memory 第2-6层都是上一层输出的output
        src_spatial_shapes: [4, 2] 4个特征层的原始shape
        src_level_start_index: [4,] 4个特征层flatten后的开始index
        src_padding_mask: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                                      tgt_key_padding_mask, tgt_reference_points, \
                                      memory, memory_key_padding_mask, memory_level_start_index, \
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                                      tgt_key_padding_mask, tgt_reference_points, \
                                      memory, memory_key_padding_mask, memory_level_start_index, \
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out
    )
