
import os
import gdown
import zipfile
from scripts import resize_youtube


LICENSE = """
These are either re-distribution of the original datasets or derivatives (through simple processing) of the original datasets.
Please read and respect their licenses and terms before use.
You should cite the original papers if you use any of the datasets.

Links:

YouTubeVOS: https://youtube-vos.org
DAVIS: https://davischallenge.org/
"""

print(LICENSE)
print('Datasets will be downloaded and extracted to ../YouTube2018, ../DAVIS16, ../DAVIS17')
reply = input('[y] to confirm, others to exit: ')
if reply != 'y':
    exit()


"""
DAVIS dataset
"""
# Google drive mirror: https://drive.google.com/drive/folders/1hEczGHw7qcMScbCJukZsoOW4Q9byx16A?usp=sharing
os.makedirs('../DAVIS/2017', exist_ok=True)

print('Downloading DAVIS 2016...')
gdown.download('https://drive.google.com/uc?id=198aRlh5CpAoFz0hfRgYbiNenn_K8DxWD', output='../DAVIS/DAVIS-data.zip', quiet=False)

print('Downloading DAVIS 2017 trainval...')
gdown.download('https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d', output='../DAVIS/2017/DAVIS-2017-trainval-480p.zip', quiet=False)

print('Downloading DAVIS 2017 testdev...')
gdown.download('https://drive.google.com/uc?id=1fmkxU2v9cQwyb62Tj1xFDdh2p4kDsUzD', output='../DAVIS/2017/DAVIS-2017-test-dev-480p.zip', quiet=False)

print('Downloading DAVIS 2017 scribbles...')
gdown.download('https://drive.google.com/uc?id=1JzIQSu36h7dVM8q0VoE4oZJwBXvrZlkl', output='../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip', quiet=False)

print('Extracting DAVIS datasets...')
with zipfile.ZipFile('../DAVIS/DAVIS-data.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/')
os.rename('../DAVIS/DAVIS', '../DAVIS/2016')

with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-trainval-480p.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
os.rename('../DAVIS/2017/DAVIS', '../DAVIS/2017/trainval')

with zipfile.ZipFile('../DAVIS/2017/DAVIS-2017-test-dev-480p.zip', 'r') as zip_file:
    zip_file.extractall('../DAVIS/2017/')
os.rename('../DAVIS/2017/DAVIS', '../DAVIS/2017/test-dev')

print('Cleaning up DAVIS datasets...')
os.remove('../DAVIS/2017/DAVIS-2017-trainval-480p.zip')
os.remove('../DAVIS/2017/DAVIS-2017-test-dev-480p.zip')
os.remove('../DAVIS/2017/DAVIS-2017-scribbles-trainval.zip')
os.remove('../DAVIS/DAVIS-data.zip')

# YouTubeVOS 2018
os.makedirs('../YouTube2018', exist_ok=True)
os.makedirs('../YouTube2018/all_frames', exist_ok=True)

print('Downloading YouTubeVOS2018 val...')
gdown.download('https://drive.google.com/uc?id=1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr', output='../YouTube2018/valid.zip', quiet=False)
print('Downloading YouTubeVOS2018 all frames valid...')
gdown.download('https://drive.google.com/uc?id=1yVoHM6zgdcL348cFpolFcEl4IC1gorbV', output='../YouTube2018/all_frames/valid.zip', quiet=False)

print('Extracting YouTube2018 datasets...')
with zipfile.ZipFile('../YouTube2018/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube2018/')
with zipfile.ZipFile('../YouTube2018/all_frames/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube2018/all_frames')

print('Cleaning up YouTubeVOS2018 datasets...')
os.remove('../YouTube2018/valid.zip')
os.remove('../YouTube2018/all_frames/valid.zip')


print('Done.')
