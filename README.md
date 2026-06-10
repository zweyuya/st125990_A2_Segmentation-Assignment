# A2 Segmentation Assignment
---

## A2-01: Object Detection — YOLOv4

### Commands Used

```bash
# Inference: YOLOv3 pretrained on dog-cycle-car.png
python3 run.py --model yolov3 --weights yolov3.weights --image dog-cycle-car.png --infer

# Train YOLOv4 on COCO (MSE/IoU loss, 5 epochs)
python3 run.py --model yolov4 --dataset coco --epochs 5 --loss mse --train

# Train YOLOv4 on COCO (CIoU loss, 5 epochs)
python3 run.py --model yolov4 --dataset coco --epochs 5 --loss ciou --train

# Evaluate mAP — YOLOv4 MSE model
python3 run.py --model yolov4 --weights yolov4_mse_epoch5.pt --dataset coco --evaluate

# Evaluate mAP — YOLOv4 CIoU model
python3 run.py --model yolov4 --weights yolov4_ciou_epoch5.pt --dataset coco --evaluate
```

### Results

| Model | Dataset | mAP@50 | Time/epoch | Notes |
|---|---|---|---|---|
| YOLOv3 (pretrained) | COCO | 0.0000 | 156.2s | inference only |
| YOLOv4 (IoU loss) | COCO | 0.0000 | 187.2s | trained from scratch |
| YOLOv4 (CIoU loss) | COCO | 0.0000 | 195.2s | loss comparison |

Model Results Files: [Click here](https://drive.google.com/drive/folders/1wU0UKfNkxLHU1KzmWKTPknRD0N7oSUzP?usp=sharing)

### Discussion

CIoU loss replaces standard MSE bounding-box regression with a geometry-aware penalty that simultaneously optimises overlap area, centre-point distance, and aspect-ratio consistency, making gradient signals more informative for poorly-aligned anchor predictions. Although both models reached mAP@50 of 0.0000 after 5 epochs (with final total losses of 4611.18 for MSE/IoU and 4226.59 for CIoU), the lower final loss of the CIoU model indicates it is converging faster and would likely outperform the MSE model given more epochs. The main challenge when training on COCO was managing the very large fraction of background anchors — the lambda_noobj weighting and gradient clipping were essential to prevent the confidence loss from overwhelming the box and classification terms. Training at high resolution required reducing the batch size to fit within GPU memory, which slowed convergence.

---

## A2-02: Image Segmentation — U-Net with ResNet-18

### Commands Used

```bash
# 1. Baseline — ResNet-18 encoder + skip connections
python3 run.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train

# 2. Ablation — same ResNet-18 encoder, skip connections REMOVED
python3 run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

# Evaluate saved model
python3 run.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate
```

### Results

| Model | Encoder | Skip connections | Val mIoU | Time/epoch |
|---|---|---|---|---|
| `unet_resnet18` | ResNet-18 (ImageNet) | ✅ | 0.7539 | 23.1s |
| `unet_resnet18_no_skip` | ResNet-18 (ImageNet) | ❌ | 0.6880 | 20.1s |


Model Results Files: [Click here](https://drive.google.com/drive/folders/1wU0UKfNkxLHU1KzmWKTPknRD0N7oSUzP?usp=sharing)

### Discussion

Skip connections improved Val mIoU by 0.0660 (from 0.6880 to 0.7539), demonstrating that preserving high-resolution spatial features in the decoder is critical for precise pixel-level segmentation. Segmentation requires both semantic understanding (what the object is) and spatial precision (exactly where its boundaries are); skip connections directly supply the fine-grained edge and texture information from shallow encoder layers that the bottleneck alone cannot reconstruct. Classification, by contrast, only needs to know *what* is in the image — a single global feature vector suffices, so skip connections provide less benefit there. U-Net is preferred over Mask R-CNN when the task is semantic segmentation (labelling every pixel by class) rather than instance segmentation (separating individual objects of the same class); U-Net is lighter, simpler to train, and generalises well with limited data when combined with a pretrained encoder.
