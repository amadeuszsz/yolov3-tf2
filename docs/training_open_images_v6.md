# Training Instruction

## Open Images Dataset V6 from Scratch

Full instruction on how to train using Open Images Dataset V6 from scratch

Requirement:
  1. Able to detect image using pretrained darknet model
  2. Many Gigabytes of Disk Space
  3. High Speed Internet Connection Preferred
  4. GPU Preferred


### 1. Download dataset & convert annotations

You can read the full description of dataset [here](https://storage.googleapis.com/openimages/web/factsfigures.html),
dataset building tool [here](https://voxel51.com/docs/fiftyone/) and annotations converting tool [here](https://github.com/chuangzhu/oidv6-to-voc).
```bash
pip3 install fiftyone  # Install tool for downloading dataset
pip3 install oidv6-to-voc  # Install csv to xml annotations converter
python3 tools/dataset_downloader.py  # Download specific dataset
oidv6-to-voc data/traffic_sign/train/labels/detections.csv -d data/traffic_sign/train/metadata/classes.csv --imgd data/traffic_sign/train/data/ --outd data/traffic_sign/train/annotations/  # Convert annotations
ls data/traffic_sign/train/data  # Explore the dataset
```

### 2. Transform Dataset

See tools/open_images_v6.py for implementation, this format is based on [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Many fields 
are not required, I left them there for compatibility with official API.

```bash
python3 tools/open_images_v6.py --data_dir 'data/traffic_sign/train/' --classes 'data/custom.names'
```

### 3. Training

You can adjust the parameters based on your setup.

#### With Transfer Learning

This step requires loading the pretrained darknet (feature extractor) weights.
```
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python3 convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny

python3 train.py \
	--dataset ./data/traffic_sign/train/signs_train.tfrecord \
	--val_dataset ./data/traffic_sign/train/signs_val.tfrecord \
	--classes ./data/custom.names \
	--num_classes 1 \
	--weights_num_classes 80 \
	--batch_size 16 \
	--epochs 100 \
	--mode fit \
	--transfer darknet \
	--weights ./checkpoints/yolov3-tiny.tf \
	--tiny
```

Original pretrained yolov3 has 80 classes, here we demonstrated how to
do transfer learning on a single class.


### 3. Inference

```bash
# detect from images
python3 detect.py \
	--classes ./data/custom.names \
	--num_classes 1 \
	--weights ./checkpoints/yolov3_train_13.tf \
	--image ./data/sign_gazebo.png \
	--tiny

# detect from validation set
python3 detect.py \
	--classes ./data/custom.names \
	--num_classes 1 \
	--weights ./checkpoints/yolov3_train_13.tf \
	--tfrecord ./data/traffic_sign/train/signs_val.tfrecord \
	--tiny
```
Weights filename may differ.


You should see some detect objects in the standard output and the visualization at `output.jpg`.
this is just a proof of concept, so it won't be as good as pretrained models.
In my experience, you might need lower score threshold if you didn't train it enough.

