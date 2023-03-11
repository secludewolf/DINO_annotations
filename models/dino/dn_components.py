# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
    """
    """
        在训练时，构造去噪任务的 queries（CW 在这里简称为加噪） 以及为它们分配标签，
        然后将这些噪声 queries 与匈牙利匹配任务的 queries 拼接（concat）起来，最后一并送入到 transformer 中一起玩。
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: 是训练还是推理
        :param num_queries: query数量 900
        :param num_classes: 类别数量 
        :param hidden_dim: 隐藏层大小 256
        :param label_enc: 编码标签 Embedding(104, 256)
        :return:
    """

    ''' 训练期间，引入去噪任务相关的部分'''
    if training:  # 如果是训练
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        # 分为积极/消极(加噪/正常)query
        dn_number = dn_number * 2

        # 计算一些索引，以便后续计算 loss 时用作 query & gt 的匹配

        # list 中的每个都是值为 1 shape 为 (num_gt_img,) 的tensor
        # 注意，每个tensor的 shape 不一定一样，因为每张图片的 目标 数量不一定一致
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        # 批量大小
        batch_size = len(known)
        # 该 batch 中各图片的目标数量
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:  # 如果有目标
            dn_number = 1
        else:  # 如果有目标
            if dn_number >= 100:
                # 一半检测积极,一半检测消极
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        ''' 准备 gt labels & gt boxes '''

        # 对 gt 在整个batch中计算索引
        # (num_gts_batch,) 其中每個值都是1
        unmask_bbox = unmask_label = torch.cat(known)
        # gt labels
        # (num_gts_batch,)
        labels = torch.cat([t['labels'] for t in targets])
        # gt boxes
        # (num_gts_batch,4)
        boxes = torch.cat([t['boxes'] for t in targets])
        # 每張圖片的 batch 索引，這個變量用於代表各圖片是第幾張圖
        # (num_gts_batch,)
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        # (num_gts_batch,1)
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        # (num_gts_batch,)
        known_indice = known_indice.view(-1)

        # “复制”到所有去噪组
        # (num_gts_batch,)->(2 * dn_number,num_gts_batch)->(2 * dn_number * num_gts_batch)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)

        # (num_gts_batch,)->(2 * dn_number * num_gts_batch,)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        # (num_gts_batch,)->(2 * dn_number * num_gts_batch,)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        # 将以上“复制”到所有去噪组
        # (num_gts_batch,4)->(2 * dn_number * num_gts_batch,4)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        # 用於在 gt labels上加噪
        known_labels_expaned = known_labels.clone()
        # 用於在 gt boxes 上加噪
        known_bbox_expand = known_bboxs.clone()

        ''' 对 gt labels 加噪 '''
        # noise on the label
        # label_noise_ratio 是用於 gt classes 的噪聲概率，默認是 0.5，即有50%的噪聲比例
        if label_noise_ratio > 0:
            # (2 * dn_number * num_gts_batch,) 從均勻分佈中採樣
            p = torch.rand_like(known_labels_expaned.float())
            # (2 * dn_number * num_gts_batch,)
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            # paper 中的 'flip' 操作，隨機選擇任意的類別作為噪聲類別
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            # 在 dim0 中使用 chosen_indice 作為 index，new_label 作為值
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        # 該 batch 中一張圖最多的 gt 數量
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)

        ''' 对 gt boxes 加噪 '''

        # noise on the box
        # box_noise_scale 是用於 gt boxes 的 scale 超參(paper 中的 lambda)，默認是 1
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            # 噪聲偏移量，作用在 gt boxes 上以實現中心點位移以及尺度縮放
            # (scalar*num_gts_batch,4)
            diff = torch.zeros_like(known_bboxs)
            # bbox 中心點坐標: w/2,h/2
            diff[:, :2] = known_bboxs[:, 2:] / 2
            # bbox 寬高: w,h
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # 在原 gt boxes 上加上偏移量，并且保证加噪后框的中心点在原来的框内
            # torch.rand_like(known_bbox_expand) * 2 - 1.0 的值域是 [-1,1)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        # 原 gt boxes 是 [0,1] 归一化的数值，于是这里进行反归一化
        # (scalar*num_gts_batch,4)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        ''' padding: 使得该 batch 中每張圖都擁有相同數量的 noised labels & noised boxes '''

        # 将以上“扩展”到所有去噪组
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        ''' 將去噪(dn)任務和匹配(matching)任務的 queries 拼接在一起 '''
        # (batch_size,pad_size + num_queries*num_patterns,hidden_dim)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        # (batch_size,pad_size + num_queries*num_patterns,4)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        ''' 
        由于以上 input_query_label & input_query_bbox 是 padded 的，
        因此要将每张图片真实有效的 noised lables(前面的 input_label_embed) & noised boxes(前面的 input_bbox_embed) 放到正确的位置上 
        '''

        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            # 将 gt 在其所在圖片中排序，以计算索引
            # 以下 List 中每个 tensor 的值域是 [0,num_gt_img-1]
            # (num_gts_batch,)
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            # 计算出去噪任务中真实有效的(非 padding 的) queries 对应的索引
            # 給每個去噪組加上一個對應的 offset，使得不同去噪組的 indices 可區分
            # i 的值域是 [0, scalar-1]，以上 map_known_indice 的值域是 [0,single_pad-1]，
            # 因此以下计算出的 map_known_indice 的值域不會超過 pad_size(即 single_pad * scalar)
            # (num_gts_batch*scalar,)
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            # 將去噪任务中真实有效的 noised lables & noised boxes “塞”到正确的位置上
            # known_pid 和 map_known_indice 的 shape 都是 (scalar*num_gts_batch)，一一對應
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        ''' 设置 attention mask 以防作弊 '''

        # 去噪任务 + 匹配任务 的 queries 总数
        tgt_size = pad_size + num_queries
        # (i,j) = True 代表 i 不可見 j
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        # 令匹配任務的 queries 看不到做去噪任務的 queries，因為後者含有真實標籤的信息
        attn_mask[pad_size:, :pad_size] = True

        # reconstruct cannot see each other
        # 对于去噪任务的 queries，只有同组内的相互可见，避免跨组泄露真實標籤的信息，
        # 因为每组中，gt 和 query 是 one-to-one 的。
        # 于是，在同一组内，对于每个 query 来说，其它 queries 都不会有自己 gt 的信息
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,  # 该 batch 中噪声 queries 的数量(包括 padding 的)
            'num_dn_group': dn_number,
        }
    else:  # 如果是推理
        # 推理时仅有原始 DETR 匹配任务的 queries
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


