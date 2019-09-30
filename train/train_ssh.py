from __future__ import print_function
import os
import sys
import argparse
import pprint
import re
import mxnet as mx
import numpy as np
from mxnet.module import Module
import mxnet.optimizer as optimizer

from utils.logger import logger
from utils.config import config, default, generate_config
from rcnn.symbol import symbol_ssh
from rcnn.core import callback, metric
from rcnn.core.loader import AnchorLoader, AnchorLoaderFPN, CropLoader
from rcnn.core.module import MutableModule
from tools.load_data import load_gt_roidb, merge_roidb, filter_roidb
from tools.load_model import load_param
from utils.utils import *

def get_fixed_params(symbol, fixed_param):
	fixed_param_names = []
	for name in symbol.list_arguments():
		for f in fixed_param:
			if re.match(f, name):
				fixed_param_names.append(name)
	return fixed_param_names

def save_model(epoch, mod, prefix):
	arg, aux = mod.get_params()
	all_layers = mod.symbol.get_internals()
	outs = []
	for stride in config.RPN_FEAT_STRIDE:
		num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
		_name = 'rpn_cls_score_stride%d_output' % stride        #将class score取出，去掉后面的loss相关的层，作为种类预测的输出
		rpn_cls_score = all_layers[_name]

		# prepare rpn data
		rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
												  shape=(0, 2, -1, 0),
												  name="rpn_cls_score_reshape_stride%d" % stride)
		rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
												   mode="channel",
												   name="rpn_cls_prob_stride%d" % stride)
		rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
												 shape=(0, 2 * num_anchors, -1, 0),
												 name='rpn_cls_prob_reshape_stride%d' % stride)
		# 将bbox和kpoint分别取出，去掉后面的loss相关的层，作为推理网络的最终输出
		rpn_bbox_pred = all_layers['rpn_bbox_pred_stride{}_output'.format(stride)]
		rpn_kpoint_pred = all_layers['rpn_kpoint_pred_stride{}_output'.format(stride)]
		outs.append(rpn_cls_prob_reshape)
		outs.append(rpn_bbox_pred)
		outs.append(rpn_kpoint_pred)
	_sym = mx.sym.Group(outs)
	mx.model.save_checkpoint(prefix, epoch, _sym, arg, aux)

def train_net(args, ctx, epoch, prefix, begin_epoch, end_epoch,
			  lr=0.001, lr_step='5'):
	# setup config
	input_batch_size = config.TRAIN.BATCH_IMAGES * len(ctx)
	# logger.info(pprint.pformat(config))

	# load dataset and prepare imdb for training
	image_sets = [iset for iset in args.image_set.split('+')]
	roidbs = [load_gt_roidb(args.dataset, image_set, args.root_path, args.dataset_path,
							flip=not args.no_flip)
			  for image_set in image_sets]
	roidb = merge_roidb(roidbs)
	roidb = filter_roidb(roidb)

	#get symbol
	sym = eval(args.symbol+".get_ssh_train")()
	sym.save('ssha_train.json')
	# mx.viz.plot_network(sym)
	#logger.info(sym.get_internals())

	feat_sym = []
	for stride in config.RPN_FEAT_STRIDE:
		feat_sym.append(sym.get_internals()['rpn_cls_score_stride%s_output' % stride])

	train_data = CropLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=not args.no_shuffle,
							ctx=ctx, work_load_list=args.work_load_list)

	# infer max shape
	max_data_shape = [('data', (1, 3, max([v[1] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
	max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)    #获取data的最大shape和其对应的label的shape
	max_data_shape.append(('gt_boxes', (1, roidb[0]['max_num_boxes'], 5)))
	logger.info('providing maximum shape %s %s' % (max_data_shape, max_label_shape))

	# infer shape
	#推断网络各个层的输出的shape的大小
	data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
	arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
	arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
	out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
	aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
	logger.info('output shape %s' % pprint.pformat(out_shape_dict))

	fixed_param_prefix = config.FIXED_PARAMS
	# load and initialize params
	if args.resume:
		arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
		fixed_param_prefix = config.FIXED_PARAMS = ['^.*upsampling']
	elif args.pretrained is not None:      #不同层的weight的初始化
		arg_params, aux_params = load_param(args.pretrained, epoch, convert=True)
		for k in ['rpn_conv_3x3', 'rpn_cls_score', 'rpn_bbox_pred', 'cls_score', 'bbox_pred']:
			_k = k + "_weight"
			if _k in arg_shape_dict:
				v = 0.001 if _k.startswith('bbox_') else 0.01
				arg_params[_k] = mx.random.normal(0, v, shape=arg_shape_dict[_k])
				logger.info('init %s with normal %.5f'%(_k,v))
			_k = k + "_bias"
			if _k in arg_shape_dict:
				arg_params[_k] = mx.nd.zeros(shape=arg_shape_dict[_k])
				logger.info('init %s with zero'%(_k))
	else:
		logger.info("arg_params = 0; aux_params = 0")
		arg_params = dict()
		aux_params = dict()
		for k, v in arg_shape_dict.items():
			if k.find('upsampling') >= 0:
				logger.info('initializing upsampling_weight', k)
				arg_params[k] = mx.nd.zeros(shape=v)
				init = mx.init.Initializer()
				init._init_bilinear(k, arg_params[k])
				#logger.info(args[k])
		fixed_param_prefix = config.FIXED_PARAMS = ['^.*upsampling']

	# create solver
	data_names = [k[0] for k in train_data.provide_data]
	label_names = [k[0] for k in train_data.provide_label]

	fixed_param_names = get_fixed_params(sym, fixed_param_prefix)
	mod = Module(sym, data_names=data_names, label_names=label_names,
				 logger=logger, context=ctx, work_load_list=args.work_load_list,
				 fixed_param_names=fixed_param_names)

	# decide training params
	# metric
	eval_metrics = mx.metric.CompositeEvalMetric()
	mids = [0, 6, 12]
	for mid in mids:
		_metric = metric.RPNAccMetric(pred_idx=mid, label_idx=mid+1)
		eval_metrics.add(_metric)
		_metric = metric.RPNL1LossMetric(loss_idx=mid+2, weight_idx=mid+3)
		eval_metrics.add(_metric)
		_metric = metric.RPNL1LossMetricKpoint(loss_idx=mid+4, weight_idx=mid+5)
		eval_metrics.add(_metric)

	# eponch callback
	# means = np.tile(np.array(config.TRAIN.BBOX_MEANS), config.NUM_CLASSES)
	# stds = np.tile(np.array(config.TRAIN.BBOX_STDS), config.NUM_CLASSES)
	#epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
	epoch_end_callback = None

	# decide learning rate
	base_lr = lr
	lr_factor = 0.8
	lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
	lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
	lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
	lr_iters = [int(epoch * len(roidb) / input_batch_size) for epoch in lr_epoch_diff]
	#lr_iters = [36000,42000] #TODO
	logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
	#lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)

	# optimizer
	opt = optimizer.SGD(learning_rate=lr, momentum=0.9, wd=0.0005, rescale_grad=1.0/len(ctx), clip_gradient=None)
	initializer = mx.init.Xavier()
	#initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style

	if len(ctx) > 1:
		train_data = mx.io.PrefetchingIter(train_data)
	_cb = mx.callback.Speedometer(train_data.batch_size, frequent=args.frequent, auto_reset=False)

	def _batch_callback(param):
		#global global_step
		_cb(param)
		global_step[0] += 1
		mbatch = global_step[0]
		save_step[0] += 1
		if save_step[0] % 5000 == 0:
			save_model(save_step[0]/5000, mod, prefix)        #保存model
			logger.info('lr: %f' % opt.lr)
		for _iter in lr_iters:    #达到一定的batch_iter之后，降低学习率
			if mbatch == _iter:
				opt.lr *= 0.8
				logger.info('lr change to', opt.lr, ' in batch', mbatch, file=sys.stderr)
				break
		#到最后一次迭代之后保存好model然后退出
		if mbatch == lr_iters[-1]:
			logger.info('saving final checkpoint', mbatch, file=sys.stderr)
			save_model(0, mod, prefix)
			sys.exit(0)

	# train
	save_step = [0]
	global_step = [0]
	mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
			batch_end_callback=_batch_callback, kvstore=args.kvstore,
			optimizer=opt,
			initializer=initializer,
			allow_missing=True,
			arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def parse_args():
	parser = argparse.ArgumentParser(description='Train Faster R-CNN network')
	# general
	parser.add_argument('--network', help='network name', default=default.network, type=str)
	parser.add_argument('--symbol', help='symbol name', default=default.symbol, type=str)
	parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
	args, rest = parser.parse_known_args()
	generate_config(args.network, args.dataset)
	parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
	parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
	parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
	# training
	parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
	parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
	parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
	parser.add_argument('--no_flip', help='disable flip images', action='store_true')
	parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
	parser.add_argument('--resume', help='continue training', default=default.resume, action='store_false')
	# e2e
	parser.add_argument('--gpus', help='GPU device to train with', default=default.gpus, type=str)
	parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
	parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
	parser.add_argument('--prefix', help='new model prefix', default=default.e2e_prefix, type=str)
	parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=default.begin_epoch, type=int)
	parser.add_argument('--end_epoch', help='end epoch of training', default=default.e2e_epoch, type=int)
	parser.add_argument('--lr', help='base learning rate', default=default.e2e_lr, type=float)
	parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default=default.e2e_lr_step, type=str)
	parser.add_argument('--no_ohem', help='disable online hard mining', action='store_true')
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	# logger.info('Called with argument: %s' % args)
	ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
	# make_if_not_exist(args.prefix)
	train_net(args, ctx, args.pretrained_epoch, args.prefix, args.begin_epoch, args.end_epoch,
			  lr=args.lr, lr_step=args.lr_step)

if __name__ == '__main__':
	main()
