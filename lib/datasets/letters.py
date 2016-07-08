#! /usr/bin/env python

#COPYRIGHT (c) 2016 OUSSAMA ENNAFII ALL RIGHTS RESERVED

import os
import errno
from datasets.imdb import imdb
import numpy as np
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import scipy.sparse
from letters_eval import letters_eval

class letters(imdb):
    def __init__(self, image_set, devkit_path):
        imdb.__init__(self, 'letters_' + image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', ',', '-', "'")
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Images', index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
	return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open( image_set_file, 'r') as f:
            image_index = map( str.strip, f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = map( self._load_letters_annotation, self.image_index)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_letters_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of Letters.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        
        with open( filename, 'r') as annot:
            lines = annot.readlines()
            lines = map( str.strip, lines) # ignore EOL
            lines = filter( lambda x: x != '', lines) # ignore empty lines
            elements = map( lambda x:x.split(' '), lines) # splitting each line
            elements = map( lambda x:filter(lambda y: y != '', x), elements) # ignoring empty entries
            elements = filter( lambda x: len(x) != 5, elements) # filtering corrupted annotation
            elements = np.array( map( lambda x: map( lambda y:float(y), x), elements) ) # str -> int to all entries
            gt_classes = np.array( elements[ :, 0] ,dtype=np.uint16)
            boxes = np.array( elements[ :, 1:], dtype = np.uint32)
            overlaps = scipy.sparse.csr_matrix( np.array( map(lambda x:list(np.eye(1,42,x)), gt_classes), dtype = np.float32))
            seg_areas = np.apply_along_axis( lambda x:(np.float32(x[1]) - np.float32(x[0]) + 1)*(np.float32(x[3])-np.float32(x[2])+1), 1, boxes)
        
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_letters_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = self._get_letters_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_letters_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_letters_results_file_template().format(cls)
                os.remove(filename)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_letters_results_file_template(self):
        # Letters_devkit/results/comp4-44503_det_test_{%s}.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        try:
            os.mkdir(self._devkit_path + '/results')
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._data_path,
            'Annotations',
            '{:s}.txt')
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_letters_results_file_template().format(cls)
            rec, prec, ap = letters_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
