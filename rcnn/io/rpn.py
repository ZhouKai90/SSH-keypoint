"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

from __future__ import print_function
import sys
import logging
import numpy as np
import numpy.random as npr

from utils.logger import logger
from utils.config import config
from .image import get_image, tensor_vstack, get_crop_image
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform, kpoint_transform, nonlinear_transform

STAT = {0:0, 8:0, 16:0, 32:0}

def get_rpn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {}

    return data, label, im_info

def get_rpn_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes}

    return data, label

def get_crop_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    #assert len(roidb) == 1, 'Single batch only'
    data_list = []
    label_list = []
    imgs, roidb = get_crop_image(roidb)
    assert len(imgs)==len(roidb)
    for i in range(len(imgs)):
      im_array = imgs[i]
      im_info = np.array([roidb[i]['im_info']], dtype=np.float32)

      # gt boxes: (x1, y1, x2, y2, cls)
      if roidb[i]['gt_classes'].size > 0:
          gt_inds = np.where(roidb[i]['gt_classes'] != 0)[0]
          gt_boxes = np.empty((roidb[i]['boxes'].shape[0], 5), dtype=np.float32)
          gt_points = np.empty((roidb[i]['points'].shape[0], 11), dtype=np.float32)
          gt_boxes[:, 0:4] = roidb[i]['boxes'][gt_inds, :]
          gt_boxes[:, 4] = roidb[i]['gt_classes'][gt_inds]
          gt_points[:, 0:10] = roidb[i]['points'][gt_inds, :]
          gt_points[:, 10] = roidb[i]['points_ind'][gt_inds]
      else:
          gt_boxes = np.empty((0, 5), dtype=np.float32)
          gt_points = np.empty((0,11), dtype=np.float32)

      data = {'data': im_array,
              'im_info': im_info}
      label = {'gt_boxes': gt_boxes, 'gt_points': gt_points}
      data_list.append(data)
      label_list.append(label)

    return data_list, label_list

def assign_anchor(feat_shape, gt_boxes, im_info, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=feat_stride, ratios=list(ratios), scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    logger.debug('anchors: %s' % base_anchors)
    logger.debug('anchor shapes: %s' % np.hstack((base_anchors[:, 2::4] - base_anchors[:, 0::4],
                                                 base_anchors[:, 3::4] - base_anchors[:, 1::4])))
    logger.debug('im_info %s' % im_info)
    logger.debug('height %d width %d' % (feat_height, feat_width))
    logger.debug('gt_boxes shape %s' % np.array(gt_boxes.shape))
    logger.debug('gt_boxes %s' % gt_boxes)

    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
    logger.debug('total_anchors %d' % total_anchors)
    logger.debug('inds_inside %d' % len(inds_inside))

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    logger.debug('anchors shape %s' % np.array(anchors.shape))

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        if config.TRAIN.RPN_FORCE_POSITIVE:
          labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if logger.level == logging.DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if logger.level == logging.DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)

    if logger.level == logging.DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        logger.debug('means %s' % means)
        logger.debug('stdevs %s' % stds)

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    if logger.level == logging.DEBUG:
        if gt_boxes.size > 0:
            logger.debug('rpn: max max_overlaps %f' % np.max(max_overlaps))
        logger.debug('rpn: num_positives %f' % np.sum(labels == 1))
        logger.debug('rpn: num_negatives %f' % np.sum(labels == 0))
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        logger.debug('rpn: num_positive avg %f' % (_fg_sum / _count))
        logger.debug('rpn: num_negative avg %f' % (_bg_sum / _count))

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_weight': bbox_weights}
    return label

def assign_anchor_fpn(feat_shape, gt_boxes, gt_points, im_info):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :return: tuple
    labels: of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    bbox_targets: of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    bbox_weights: mark the assigned anchors
    kpoint_targets: of shape (batch_size, num_anchors * 10, feat_height, feat_width)
    kpoint_weights: mark the assigned anchors
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    global STAT
    DEBUG = False
    im_info = im_info[0]                            #获取输入图像的大小和crop图像针对原图的缩放比例
    # clean up boxes 去除掉其中无效的bbox标定
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    gt_points = gt_points[nonneg]
    #scales = np.array(scales, dtype=np.float32)
    feat_strides = config.RPN_FEAT_STRIDE

    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        sstride = str(stride)
        #从预先定义的配置文件中获取对应的参数
        base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']                 #base_size为啥都要为16，是因为这样才能覆盖到大中小的人脸么？
        allowed_border = config.RPN_ANCHOR_CFG[sstride]['ALLOWED_BORDER']       #RP不能超出边框
        ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
        scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
        #根据scales和ratios的值，对base_size进行扩展．获得RP的值.每个anchors对应的RP个数为len(ratios)^len(scales)
        base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32))
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shape[i][-2:]
        feat_stride = feat_strides[i]
        feat_infos.append([feat_height, feat_width])

        A = num_anchors
        A_list.append(A)    #将不同stride下每个anchors生成的对应原图的RP个数保存下来
        K = feat_height * feat_width        #不同stride下得到的rpn_feature的大小，即anchors的个数

        #生成所有的anchors对应的RP坐标
        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))      #矩阵变换

        total_anchors = int(K * A)
        anchors_num_list.append(total_anchors)               #记录每个stride下生成的RP总个数
        # only keep anchors inside the image
        #去除掉已经超出的RP
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        #print('AA', anchors.shape, len(inds_inside))

        anchors_list.append(anchors)
        inds_inside_list.append(inds_inside)

    # Concat anchors from each level
    anchors = np.concatenate(anchors_list)              #将3个不同stride下产生的RP合并起来，所有RP的个数大约为(20*20+40*40+80*80)*2，当然还要去掉其中超出边框的部分
    for i in range(1, len(inds_inside_list)):
        inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])   #将index也合并一下，由[[0,1,2,...], [0,1,2,...], [0,1,..]]变成[0,1,...,800, 801..., 1600,...]
    inds_inside = np.concatenate(inds_inside_list)
    total_anchors = sum(anchors_num_list)
    #print('total_anchors', anchors.shape[0], len(inds_inside), file=sys.stderr)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    #print('BB', anchors.shape, len(inds_inside))
    #print('gt_boxes', gt_boxes.shape, file=sys.stderr)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))   #将每个RP坐标和gt boxes进行IOU计算
        argmax_overlaps = overlaps.argmax(axis=1)   #argmaxu()返回最大值索引．得到每个RP和原图中多个gt_boxes中的哪一个的IOU值最大的索引
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]   #取出inside中，每个RP和多个gt＿bbox中IOU最大值　
        gt_argmax_overlaps = overlaps.argmax(axis=0)    #得到和每个gt_bbox　IOU 最大的RP的索引
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]    #取得和每个gt_bbox　IOU最大值
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]               #获取满足上一步条件的RP的索引

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0            #将IOU值小于0.3的RP设置为positive

        # fg label: for each gt, anchor with highest overlap
        #对于论文中的两种挑选正样本的方法，１：选IOU最大的(会非常少)　2:选IOU超过某个阈值的
        if config.TRAIN.RPN_FORCE_POSITIVE:                                         #将IOU最大的RP设置为positive．
          labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1               #选IOU超过某个阈值的

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0
    fg_inds = np.where(labels == 1)[0]
    #print('fg count', len(fg_inds))

    # subsample positive labels if we have too many
    if config.TRAIN.RPN_ENABLE_OHEM==0:
      fg_inds = np.where(labels == 1)[0]
      num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
      if len(fg_inds) > num_fg:
          disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
          if DEBUG:
              disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
          labels[disable_inds] = -1

      # subsample negative labels if we have too many
      num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
      bg_inds = np.where(labels == 0)[0]
      if len(bg_inds) > num_bg:
          disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
          if DEBUG:
              disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
          labels[disable_inds] = -1
    else:
      fg_inds = np.where(labels == 1)[0]
      num_fg = len(fg_inds)
      bg_inds = np.where(labels == 0)[0]
      num_bg = len(bg_inds)


    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)        #根据inside的RP个数初始化对应的bbox
    kpoint_targets = np.zeros((len(inds_inside), 10), dtype=np.float32)     #根据inside的RP个数初始化对应的kpoint
    if gt_boxes.size > 0:
        bbox_targets[:] = nonlinear_transform(anchors, gt_boxes[argmax_overlaps, :4])   #将每个RP和其对应的最大IOU值的gt做回归计算,得到target
        kpoint_targets[:] = kpoint_transform(anchors, gt_points[argmax_overlaps, :10])  #以RP的中心点为参考点，分别和key_point的五个点做回归计算，得到target，这个还需要好好理解．

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)      #将正样本的bbox_weight全部赋值为1．负样本的0

    kpoint_weights = np.zeros((len(inds_inside), 10), dtype=np.float32)
    kpoint_weights[labels ==1, :] = np.array(config.TRAIN.RPN_KPOINT_WEIGHTS)   #将正样本的kpoint_weights全部赋值为1
    if gt_points.size > 0:
        gt_points_flag = gt_points[argmax_overlaps, 10]
        gt_points_flag = np.array(gt_points_flag, np.int)
        kpoint_weights[gt_points_flag == 0, :] = np.array(config.TRAIN.RPN_KPOINT_WEIGHTS_NON)      #将负样本的kpoint_weights赋值为0

    #if DEBUG:
    #    _sums = bbox_targets[labels == 1, :].sum(axis=0)
    #    _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
    #    _counts = np.sum(labels == 1)
    #    means = _sums / (_counts + 1e-14)
    #    stds = np.sqrt(_squared_sums / _counts - means ** 2)
    #    print 'means', means
    #    print 'stdevs', stds
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)             #labels表示所有样本的属性　1:正样本　0:负样本　-1:超出边框
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)     #将超出边框的RP也标记为0
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)
    kpoint_targets = _unmap(kpoint_targets, total_anchors, inds_inside, fill=0)
    kpoint_weights = _unmap(kpoint_weights, total_anchors, inds_inside, fill=0)
    #print('CC', anchors.shape, len(inds_inside))

    #if DEBUG:
    #    if gt_boxes.size > 0:
    #        print 'rpn: max max_overlaps', np.max(max_overlaps)
    #    print 'rpn: num_positives', np.sum(labels == 1)
    #    print 'rpn: num_negatives', np.sum(labels == 0)
    #    _fg_sum = np.sum(labels == 1)
    #    _bg_sum = np.sum(labels == 0)
    #    _count = 1
    #    print 'rpn: num_positive avg', _fg_sum / _count
    #    print 'rpn: num_negative avg', _bg_sum / _count

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    kpoint_target_list = list()
    kpoint_weight_list = list()
    anchors_num_range = [0] + anchors_num_list
    label = {}
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        _label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]   #将不同stride下的labels文件分离开
        #_fg_inds = np.where(_label == 1)[0]
        #n_fg = len(_fg_inds)
        #STAT[0]+=1
        #STAT[stride]+=n_fg
        #if STAT[0]%100==0:
        #  print('rpn_stat', STAT, file=sys.stderr)
        bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]

        kpoint_target = kpoint_targets[sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]
        kpoint_weight = kpoint_weights[sum(anchors_num_range[:i + 1]):sum(anchors_num_range[:i + 1]) + anchors_num_range[i + 1]]
        #维度转换
        _label = _label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        _label = _label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * 4)).transpose(0, 2, 1)
        bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * 4)).transpose((0, 2, 1))
        kpoint_target = kpoint_target.reshape((1, feat_height * feat_width, A * 10)).transpose(0, 2, 1)
        kpoint_weight = kpoint_weight.reshape((1, feat_height * feat_width, A * 10)).transpose((0, 2, 1))
        label['label_stride%d'%stride] = _label
        label['bbox_target_stride%d'%stride] = bbox_target
        label['bbox_weight_stride%d'%stride] = bbox_weight
        label['kpoint_target_stride%d'%stride] = kpoint_target
        label['kpoint_weight_stride%d'%stride] = kpoint_weight
        #print('in_rpn', stride,_label.shape, bbox_target.shape, bbox_weight.shape, file=sys.stderr)
        label_list.append(_label)
        #print('DD', _label.shape)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)
        kpoint_target_list.append(kpoint_target)
        kpoint_weight_list.append(kpoint_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)
    kpoint_target_concat = np.concatenate(kpoint_target_list, axis=2)
    kpoint_weight_concat = np.concatenate(kpoint_weight_list, axis=2)
    #fg_inds = np.where(label_concat[0] == 1)[0]
    #print('fg_inds_in_rpn2', fg_inds, file=sys.stderr)

    label.update({'label': label_concat,
            'bbox_target': bbox_target_concat,
            'bbox_weight': bbox_weight_concat,
            'kpoint_target': kpoint_target_concat,
            'kpoint_weight': kpoint_weight_concat}
            )
    return label

