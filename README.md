# ddp_test_
- Distributed Data Parallel에 익숙해지기 위한 프로젝트
1. DDP의 전반적인 흐름을 알기위해 그렇게 Deep하지 않은 간단한 모델 실험
  - Auto-encoder
    - net/ddp/autoencoder.py
    - util/ddp/dataloader/autoencoder_dataloader.py
2. 기존에 pytorch로 구현했었던 MaskRCNN을 pytorch lightning으로 옮기는 겸 DDP를 적용
  - MaskRCNN
    - net/ddp/maskrcnn.py
    - util/ddp/dataset/maskrcnndataset.py
    - util/ddp/dataloader/maskrcnn_dataloader.py
3. Attention or Transformer 모델 구현 예정..
# MLOps
- WandB 적용
  - 우선 log 및 multiple run간의 비교를 위해 적용
  - hyperparameter tuning은 추후 적용
- ONNX  
  - 다양한 machine에서 돌아갈 수 있도록 ONNX format의 모델도 출력
# 자세한 내용 설명
- DDP 관련 내용 및 개인 프로젝트 정리
- https://engui-note.tistory.com/
