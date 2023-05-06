import boto3
import cv2
import jsonlines
import numpy as np
from functools import partial
from pathlib import Path
from tqdm import tqdm

from qdtrack.apis import inference_model, init_model

dataset = "DETRAC"


class MEVASensor:

    def __init__(self, data_path, eager=True):
        cap = cv2.VideoCapture(str(data_path))
        ret = True
        self.eager = eager
        self.released = False
        if self.eager:
            self.all_data = []
            ret, frame = cap.read()
            while ret:
                self.all_data.append({"center_camera_feed": frame})
                ret, frame = cap.read()
            cap.release()
        else:
            self.cap = cap
            self.next_frame = 0

    def total_num_frames(self):
        if self.eager:
            return len(self.all_data)
        else:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, frame_index):
        if self.eager:
            return self.all_data[frame_index]
        else:
            assert self.next_frame == frame_index, (self.next_frame,
                                                    frame_index)
            self.next_frame += 1
            ret, frame = self.cap.read()
            assert ret, frame_index
            if frame_index == self.total_num_frames() - 1:
                self.cap.release()
                self.released = True
            return {"center_camera_feed": frame}

    def __del__(self):
        if not self.released:
            self.cap.release()


class DETRACSensor:

    def __init__(self, data_path, eager=True):
        data_path = Path(data_path)
        self.all_jpgs = sorted(
            data_path.glob("*.jpg"),
            key=lambda x: int(x.stem.replace("img", "")))
        if eager:
            self.frames = [
                cv2.imread(str(x)).astype(np.uint8) for x in self.all_jpgs
            ]
        else:
            self.frames = {}
        self.eager = eager

    def total_num_frames(self):
        return len(self.all_jpgs)

    def get_frame(self, frame_index):
        if self.eager or frame_index in self.frames:
            return {"center_camera_feed": self.frames[frame_index]}
        else:
            self.frames[frame_index] = cv2.imread(
                self.all_jpgs[frame_index]).astype(np.uint8)
            return {"center_camera_feed": self.frames[frame_index]}


def sync_from_aws_s3(bucket, base_path, path):
    print(base_path, path)
    path = str(path)
    if (Path(base_path) / path).exists():
        return
    print(f"Downloading {path} from AWS S3...")
    p = Path(base_path) / path
    to_make = p.parent
    to_make.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, path, str(Path(base_path) / path))


bdd100k_classes = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle',
                   'motorcycle', 'train')

bdd100k_to_waymo = {
    "pedestrian": "pedestrian",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "traffic sign": "sign"
}

qd_track_config_path = "/home/eyal/shared_with_host/qdtrack/configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py"
qd_track_model_path = "/home/eyal/shared_with_host/qdtrack/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth"

model = init_model(
    qd_track_config_path,
    checkpoint=qd_track_model_path,
    device='cuda:0',
    cfg_options=None)

avi_files = [
    'drops-123-r13/2018-03-05/09/2018-03-05.09-49-37.09-50-00.school.G474.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-49-41.09-50-01.school.G339.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-49-44.09-50-00.school.G424.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-49-46.09-50-00.school.G419.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-49-50.09-50-00.school.G420.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-49-52.09-50-00.school.G336.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-00.09-50-14.school.G420.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-00.09-50-16.school.G419.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-00.09-55-00.school.G336.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-00.09-55-00.school.G424.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-00.09-55-00.school.G474.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-01.09-55-01.school.G339.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-07.09-55-00.school.G300.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-11.09-50-14.school.G423.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-15.09-55-01.school.G423.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-16.09-55-00.school.G419.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-16.09-55-02.school.G420.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-19.09-55-01.school.G421.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-30.09-55-00.school.G299.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-33.09-55-00.school.G330.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-38.09-51-02.school.G328.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-51-04.09-55-00.school.G328.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G299.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G300.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G328.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G330.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G336.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G419.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G424.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G474.r13.avi'
]

avi_files = [
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-00.09-55-00.school.G424.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-01.09-55-01.school.G339.r13.avi',
    'drops-123-r13/2018-03-05/09/2018-03-05.09-50-15.09-55-01.school.G423.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-00.10-00-00.school.G424.r13.avi',
    'drops-123-r13/2018-03-05/10/2018-03-05.09-55-01.10-00-01.school.G339.r13.avi',
    'drops-123-r13/2018-03-11/14/2018-03-11.14-00-01.14-05-00.school.G336.r13.avi',
    'drops-123-r13/2018-03-11/16/2018-03-11.16-20-08.16-25-08.hospital.G436.r13.avi'
]

MEVA_scenario_to_path = {Path(p).name: p for p in avi_files}

DETRAC_scenario_to_path = {
    **{
        "train-" + p.stem: Path(*p.parts[2:])
        for p in Path("../ad-config-search/DETRAC/Insight-MVT_Annotation_Train").glob("*")
    },
    **{
        "test-" + p.stem: Path(*p.parts[2:])
        for p in Path("../ad-config-search/DETRAC/Insight-MVT_Annotation_Test").glob("*")
    }
}

scenario_to_path = {
    "MEVA": MEVA_scenario_to_path,
    "DETRAC": DETRAC_scenario_to_path
}

for scenario, path in scenario_to_path[dataset].items():
    base_path = "../ad-config-search"
    if dataset in ["MEVA"]:
        sync_from_aws_s3("mevadata-public-01", base_path, path)
    local_path = Path(base_path) / path
    reader = {
        "DETRAC": DETRACSensor,
        "MEVA": partial(MEVASensor, eager=False)
    }[dataset]
    reader = reader(local_path)

    num_frames = reader.total_num_frames()

    preds = []

    print(scenario)

    for frame_id in tqdm(range(num_frames)):
        frame = reader.get_frame(frame_id)["center_camera_feed"]
        results = inference_model(model, frame, frame_id)

        bbox_result, track_result = results.values()

        obstacles = []
        for k, v in track_result.items():
            track_id = k
            bbox = v['bbox']
            score = bbox[4]
            label_id = v['label']
            bdd100k_label = bdd100k_classes[label_id]
            # skip labels that don't appear on Waymo
            if bdd100k_label not in bdd100k_to_waymo:
                continue
            waymo_label = bdd100k_to_waymo[bdd100k_label]
            obstacles.append({
                "bbox": {
                    "xmn": float(bbox[0]),
                    "xmx": float(bbox[2]),
                    "ymn": float(bbox[1]),
                    "ymx": float(bbox[3])
                },
                "label": waymo_label,
                "id": int(track_id),
                "confidence": float(score)
            })
        preds.append(obstacles)

    ms_between_frames = {"MEVA": 33, "DETRAC": 40}[dataset]

    with jsonlines.open(scenario + ".jsonl", mode='w') as writer:
        writer.write({"content_type": "gt_metadata"})
        for i, obstacles in enumerate(preds):
            writer.write({
                "content_type": "ground_truth",
                "runtime": 0,
                "obstacles": obstacles,
                "timestamp": i * ms_between_frames
            })
