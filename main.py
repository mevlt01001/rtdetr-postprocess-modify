from ultralytics import RTDETR
import torch, torchvision
import json, time

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h, _= boxes.unbind(1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=1)

def gt_boxes(gtboxes):
    _boxes = []
    for boxes in gtboxes:
        x1, y1, w, h = boxes["vbox"]
        cx, cy = x1 + w/2, y1 + h/2
        _boxes.append([int(cx), int(cy), int(w), int(h)])
    return _boxes

with open("CrowdHuman_val/annotation_person.odgt", "r") as f:
    annotations = [json.loads(line) for line in f]

model = RTDETR('rtdetr-l.pt').model.to(device=torch.device("cuda"), dtype=torch.float16) 
size = (640, 640)
threshold = 0.5

file = open("results_fp16.odgt", "w")
for i,annotation in enumerate(annotations):
    # if i == 10:
    #     break
    print(f"({i+1}/4370) Processing {annotation['ID']}")

    # clear memory
    if i % 50 == 0:
        torch.cuda.empty_cache()

    # read image
    image = torchvision.io.read_image(f"CrowdHuman_val/Images/{annotation['ID']}.jpg").to(device=torch.device("cuda"), dtype=torch.float16)
    real_height, real_width = image.shape[1], image.shape[2]

    start = time.time()
    # PRE-PROCESS
    image = image/255
    image = torchvision.transforms.functional.resize(image, size).unsqueeze(0)
    pre_precess_time = (time.time() - start)*1000

    # INFERENCE
    start = time.time()
    with torch.no_grad():
        output = model.forward(image)
    inference_time = (time.time() - start)*1000

    start = time.time()
    # POST-PROCESS
    #define boxes and confs
    boxes = output[0][0][:, :4]#cxcywh
    confs = output[0][0][:, 4]
    boxes = boxes[confs >= threshold]

    #adjust to original shape
    boxes[:, 0] *= real_width
    boxes[:, 1] *= real_height
    boxes[:, 2] *= real_width
    boxes[:, 3] *= real_height
    post_process_time = (time.time() - start)*1000

    #get_result

    pred_boxes = boxes.int().tolist()
    truth_boxes = gt_boxes(annotation["gtboxes"])

    line = {
        "ID": annotation["ID"],
        "truth_boxes": truth_boxes,
        "pred_boxes": pred_boxes,
        "pre_process_time": pre_precess_time,
        "inference_time": inference_time,
        "post_process_time": post_process_time
    }

    file.write(json.dumps(line) + "\n")

file.close()