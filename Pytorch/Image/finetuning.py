# import os
# import numpy as np
# import torch
# from PIL import Image
# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision import transforms as T
# from engine import train_one_epoch, evaluate
# import utils

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
from torchvision import transforms as T

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        ## 모든 이미지 파일들 읽고, 정렬하여 이미지와 분할 마스크 정렬 확인
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        ## 이미지와 마스크 읽어오기
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        ## 분할 마스크는 RGB로 변환되지 않음
        ## 왜냐하면 각 색상은 다른 인스턴스에 해당, 0은 배경에 해당
        mask = Image.open(mask_path)
        ## numpy 배열을 PIL 이미지로 변환
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        ## 각 마스크의 바운딩 박스 좌표 얻기
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        ## 모든 것을 torch.Tensor 타입으로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        ## 모든 인스턴스는 crowd 상태가 아님을 가정
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)
        
## Torchvision model zoo(미리 학습된 모데들)에서 모델을 수정하려면 보통 두 가지 상황
## 1. 미리 학습된 모델로부터 미세 조정
## 2. 모델의 백본을 다른 백본으로 교체

# # 미리 학습된 모델 로드하고 특징들만 리턴
# backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
# # faster RCNN은 백본의 출력 채널 수를 알야 함
# # mobilenetV2의 경우 1280이므로 추가
# backbone.out_channels = 1280

# # RPN(Region Proposal Network)이 5개의 서로 다른 크기와 3개의 다른 측면 비율을 가진
# # 5 * 3개의 앵커를 공간 위치마다 생성하도록
# # 각 특징 map이 잠재적으로 다른 사이즈와 측면 비율을 가질 수 있기 때문에 tuple[tuple[int]]타입으로
# anchor_generator = AnchorGenerator(s  izes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

# # 관심 영역의 자르기 및 재할당 후 자르기 크기를 수행하는 데 사용할 feature map 정의
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# # 조각들을 Faster RCNN 모델로 합친다
# model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

## 우리의 경우 dataset이 작기 떄문에 1번 접근법
def get_model_instance_segmentation(num_classes):
    # COCO에서 미리 학습된 instance segmentation model 읽어오기
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 분류를 위한 input feature dimension 얻기
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꾸기
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 마스크 분류기를 위한 입력 특징들의 차원 얻기
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 마스크 예측기를 새로운 것으로 바꾸기
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

## 모든 것을 하나로 합치기
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # 학습시 50% 확률로 학습 영상을 좌우 반전 변환
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

## forward 메소드 테스트 = 데이터셋 반복 전에 샘플 데이터로 학습과 추론 시 모델이 예상대로 동작하는지 확인
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
# )
# # 학습 시
# images, targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images, targets)

# # 추론 시
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500 ,400)]
# predictions = model(x)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 이번 데이터셋 은 두 개의 클래스만 존재 - 배경 or 사람
    num_classes = 2
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    # dataset = PennFudanDataset('C:\\Users\\joeda\\Desktop\\python\\Pytorch\\Image\\PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    # dataset_test = PennFudanDataset('C:\\Users\\joeda\\Desktop\\python\\Pytorch\\Image\\PennFudanPed', get_transform(train=False))


    # 데이터셋을 학습용과 테스트용으로 나누기
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # data loader를 학습용과 검증용으로 정의
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    # optimizer 생성
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 학습률 스케쥴러 생성
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        # 1 epoch 동안 학습, 10회 마다 출력
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # 학습률 업데이트
        lr_scheduler.step()
        # 테스트 데이터셋에서 평가
        evaluate(model, data_loader_test, device=device)
    
    print("That's it!")

if __name__ == "__main__":
    main()