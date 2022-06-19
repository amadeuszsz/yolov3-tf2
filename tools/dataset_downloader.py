import os
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz

package_path = Path(__file__).parent.parent

dataset = foz.load_zoo_dataset(
              "open-images-v6",
              split="train",
              label_types=["detections"],
              classes=["Traffic sign"],
              dataset_dir=os.path.join(package_path, "data/traffic_sign")
)

session = fo.launch_app(dataset)
session.wait()