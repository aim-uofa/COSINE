import os
import json
import shutil

coco_imgs = [
    # '000000240754',
    # '000000077855',
    # '000000269196',
    # '000000535578',
    # '000000149221',
    # '000000536073',
    # '000000299553',
    # '000000469192'
]

ade_imgs = [
    'ADE_val_00001164', 'ADE_val_00000157',
    'ADE_val_00000213', 'ADE_val_00001720',
    'ADE_val_00000160', 'ADE_val_00001130',
]

# ref seg
maybe_coco = [
    # '000000008300',
    # '000000044788',
    # '000000009185',
    # '000000043655',
    # '000000450735',
    # '000000177529',
    # '000000127056',
    # '000000383372',
    '000000494248'
]

# openocv
mabbe_ade = [
    'ADE_val_00000703',
    'ADE_val_00001884',
    'ADE_val_00000146',
    'ADE_val_00000087',
    'ADE_val_00000243',
    'ADE_val_00000316'
]

save_root = 'outputs/img_license'
if not os.path.exists(save_root):
    os.makedirs(save_root)


coco_train_json = 'datasets/coco/annotations/instances_train2017.json'
coco_val_json = 'datasets/coco/annotations/instances_val2017.json'

ade_train_json = 'datasets/coco/annotations/instances_train2017.json'
ade_val_json = 'datasets/coco/annotations/instances_val2017.json'



with open(coco_train_json, 'r') as file:
    coco_train = json.load(file)

with open(coco_val_json, 'r') as file:
    coco_val = json.load(file)

license_ = coco_val['licenses']
print(license_)

coco_train_img_license = {}
coco_val_img_license = {}

# for img in coco_train['images']:
#     coco_train_img_license[img['file_name']] = img['license']
for img in coco_val['images']:
    coco_val_img_license[img['file_name']] = img['license']


for img_name in maybe_coco:
    if img_name+'.jpg' in coco_train_img_license:
        source = f'datasets/coco/train2017/{img_name}.jpg'
        destination = save_root + f"/{img_name}_{coco_train_img_license[img_name+'.jpg']}.jpg"
        shutil.copy(source, destination)

    elif img_name+'.jpg' in coco_val_img_license:
        source = f'datasets/coco/val2017/{img_name}.jpg'
        destination = save_root + f"/{img_name}_{coco_val_img_license[img_name+'.jpg']}.jpg"
        shutil.copy(source, destination)

    else:
        print(f'not find {img_name} !')