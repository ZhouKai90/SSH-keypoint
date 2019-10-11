import cv2
import sys
import numpy as np
import datetime
import argparse
import os
from ssh_detector import SSHDetector

# scales = [1200, 1600]
scales = [1080, 1920]

def main(args):
	detector = SSHDetector(args.prefix, args.epoch, args.gpuId)

	imgDir = args.imagesDir
	savePath = args.savePath
	imgNameList = os.listdir(imgDir)

	for imgName in imgNameList:
		imgPath = imgDir + imgName
		if not os.path.isfile(imgPath):
			continue
		print(imgPath)

		img = cv2.imread(imgPath)
		im_shape = img.shape
		print(im_shape)
		target_size = scales[0]
		max_size = scales[1]
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])
		if im_size_min > target_size or im_size_max > max_size:
			im_scale = float(target_size) / float(im_size_min)
			# prevent bigger axis from being more than max_size:
			if np.round(im_scale * im_size_max) > max_size:
				im_scale = float(max_size) / float(im_size_max)
			img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
			print('resize to', img.shape)

		# timea = datetime.datetime.now()
		faces = detector.detect(img, threshold=0.8)
		# timeb = datetime.datetime.now()

		for num in range(faces.shape[0]):
			bbox = faces[num, 0:4]
			cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
			kpoint = faces[num, 5:15]
			for knum in range(5):
				cv2.circle(img, (kpoint[2 * knum], kpoint[2 * knum + 1]), 2, [0, 0, 255], 2)
		cv2.imwrite(savePath + imgName, img)

		# diff = timeb - timea
		# print('detection uses', diff.total_seconds(), 'seconds')
		# print('find', faces.shape[0], 'faces')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuId', type=int, default=0, help='gpu id')
    parser.add_argument('--prefix', default=os.path.join(os.getcwd(), 'model/symbol_ssh'))
    parser.add_argument('--epoch', type=int, default=0, help='model to test with')
    parser.add_argument('--dataShape', default=(640, 640), type=int)
    parser.add_argument('--savePath', default=os.path.join(os.getcwd(), 'images/output/'))
    parser.add_argument('--imagesDir', default=os.path.join(os.getcwd(), 'images/ori/'))
    return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	main(args)