"""Generate a COCO-format JSON with person bboxes using mmdet.

Output JSON is directly usable as --json-file for:
  demo/body3d_two_stage_img_demo.py
  demo/mesh_img_demo.py
  demo/top_down_img_demo.py
"""
import argparse
import json
import os

import mmcv
from PIL import Image, ImageDraw

try:
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    raise ImportError('mmdet is required. Run: pip install mmdet>=2.14.0')


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
PERSON_CAT_ID = 1  # COCO category id for "person"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run mmdet on an image directory and save COCO-format JSON')
    parser.add_argument('det_config', help='mmdet config file')
    parser.add_argument('det_checkpoint', help='mmdet checkpoint file')
    parser.add_argument('img_root', help='Directory containing input images')
    parser.add_argument('out_json', help='Output JSON file path')
    parser.add_argument(
        '--bbox-thr', type=float, default=0.3,
        help='Minimum confidence score for person bbox (default: 0.3)')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--out-img-dir', default=None,
        help='Directory to save visualized bbox images (optional)')
    return parser.parse_args()


def collect_images(img_root):
    paths = []
    for fname in sorted(os.listdir(img_root)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTENSIONS:
            paths.append(fname)
    return paths


def _load_detector(config_path, checkpoint, device):
    """Work around mmdet assertion: test_cfg must not appear in both outer
    config field and model field simultaneously (older configs do both)."""
    cfg = mmcv.Config.fromfile(config_path)
    # If the model already carries test_cfg, drop the redundant outer field.
    if cfg.get('test_cfg') is not None and cfg.model.get('test_cfg') is not None:
        cfg.pop('test_cfg')
    # mmdet v2.x does not support data_preprocessor (added in v3.x).
    cfg.model.pop('data_preprocessor', None)
    # mmdet v2.x inference_detector requires cfg.data.test.pipeline.
    if not cfg.get('data') or not cfg.data.get('test'):
        cfg.data = dict(test=dict(pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize',
                         mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375],
                         to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ],
            ),
        ]))
    return init_detector(cfg, checkpoint, device=device)


def main():
    args = parse_args()

    det_model = _load_detector(
        args.det_config, args.det_checkpoint, args.device)

    fnames = collect_images(args.img_root)
    if not fnames:
        raise RuntimeError(f'No images found in {args.img_root}')
    print(f'Found {len(fnames)} images.')

    if args.out_img_dir:
        os.makedirs(args.out_img_dir, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0

    for img_id, fname in enumerate(mmcv.track_iter_progress(fnames), start=1):
        full_path = os.path.join(args.img_root, fname)
        w, h = Image.open(full_path).size

        images.append({
            'id': img_id,
            'file_name': fname,
            'width': w,
            'height': h,
        })

        # mmdet returns a list of arrays, one per class (0-indexed)
        # COCO person is class index 0 in most person-detector configs
        result = inference_detector(det_model, full_path)

        # result[0] = person class scores shaped (N, 5): x1,y1,x2,y2,score
        person_bboxes = result[0] if isinstance(result, (list, tuple)) else result

        if args.out_img_dir:
            vis_img = Image.open(full_path).convert('RGB')
            draw = ImageDraw.Draw(vis_img)

        for bbox in person_bboxes:
            x1, y1, x2, y2, score = float(bbox[0]), float(bbox[1]), \
                                     float(bbox[2]), float(bbox[3]), float(bbox[4])
            if score < args.bbox_thr:
                continue
            bw = x2 - x1
            bh = y2 - y1
            if args.out_img_dir:
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1, y1 - 12), f'{score:.2f}', fill='red')
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': PERSON_CAT_ID,
                'bbox': [round(x1, 2), round(y1, 2),
                         round(bw, 2), round(bh, 2)],  # xywh format
                'area': round(bw * bh, 2),
                'iscrowd': 0,
                'score': round(score, 4),
            })
            ann_id += 1

        if args.out_img_dir:
            vis_img.save(os.path.join(args.out_img_dir, fname))

    coco_json = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': PERSON_CAT_ID, 'name': 'person'}],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(coco_json, f, indent=2)

    print(f'\nSaved {len(images)} images, {len(annotations)} person bboxes')
    print(f'-> {args.out_json}')


if __name__ == '__main__':
    main()
