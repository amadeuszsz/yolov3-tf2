import time
import os
from pathlib import Path
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

package_path = Path(__file__).parent.parent

flags.DEFINE_string('data_dir', os.path.join(package_path, 'data/traffic_sign/train/'),
                    'path to raw dataset')
flags.DEFINE_string('classes', os.path.join(package_path, 'data/custom.names'), 'classes file')
flags.DEFINE_float('dataset_split', 0.7, 'training split ratio', 0.1, 0.95)


def build_example(annotation, class_map):
    img_path = os.path.join(
        FLAGS.data_dir, 'data', annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            if not obj['name'] in class_map:
                continue
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    path = os.path.join(FLAGS.data_dir, 'annotations')
    image_list = [file.split(".")[0] for file in sorted(os.listdir(path))]
    train_image_list = image_list[:int(len(image_list) * 0.7)]
    val_image_list = image_list[int(len(image_list) * 0.7):]
    image_list_splits = {"train": train_image_list, "val": val_image_list}
    logging.info("Image list loaded: %d", len(image_list))

    for key in image_list_splits:
        writer = tf.io.TFRecordWriter(FLAGS.data_dir + f"signs_{key}.tfrecord")
        for name in tqdm.tqdm(image_list_splits[key]):
            annotation_xml = os.path.join(
                FLAGS.data_dir, 'annotations', name + '.xml')
            annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
            annotation = parse_xml(annotation_xml)['annotation']
            tf_example = build_example(annotation, class_map)
            writer.write(tf_example.SerializeToString())
        writer.close()
        logging.info("Done")


if __name__ == '__main__':
    app.run(main)
