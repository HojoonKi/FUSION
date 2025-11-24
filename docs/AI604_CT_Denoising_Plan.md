# AI604 CT Denoising Roadmap

## 1. 참고 자료 요약
- **제안서(`_AI604__CT_Denoising_Proposal.pdf`) 핵심**  
  - LDCT → NDCT 복원 시 구조 정보 손실 및 노이즈 제거 사이 균형 문제.  
  - 저주파 구조 정보를 보존하기 위해 FFT 기반 구조 프라이어를 만들고, 이를 이미지 도메인의 디퓨전 U-Net에 **크로스 어텐션**으로 주입하는 FUSION 아키텍처 제안.  
  - 조건부 DDPM/DDIM 백본 + 저주파 지도(low-pass filtered image) + 단계별 구조 보강이 목표.
- **데이터 전처리 노트북(`Data_Preparation.ipynb`)**  
  - Google Drive 상 LDCT/NDCT DICOM(`QD_1mm/FD_1mm`)을 압축 해제 후 슬라이스 단위로 읽어 HU 변환.  
  - `datasets` 라이브러리 스키마(환자, slice index, qd, fd)를 정의하고 train/test 세트를 `ct_denoise_{train,test}` 디렉터리와 zip으로 저장.  
  - 향후 로컬 학습 시 동일 구조(`data/ct_denoise_{train,test}/dataset_info.json ...`)를 그대로 배치하거나, PNG 변환 시 `_ld.png`, `_fd.png` 쌍이 되도록 `fusion_datasets/LDFDCT.py` 스캐닝 규칙에 맞춰야 함.

## 2. 현재 자산 정리
| 항목 | 내용 | TODO |
| --- | --- | --- |
| 데이터 | 노트북 기반으로 생성된 `ct_denoise_train/test` (Hugging Face datasets 형식)과 PNG Pair(`*_ld.png`, `*_fd.png`) 폴더가 Drive에 존재 | 로컬 `data/` 아래 동일 구조 생성 후 `configs/*/train_dataroot` 업데이트 |
| 전처리 코드 | `Data_Preparation.ipynb` (압축 해제, HU 변환, HF Dataset 생성, 시각화) | 로컬/서버용 `.py` 스크립트화 고려 (선택) |
| 모델 | 기존 Fast-DDPM + Fourier 모듈 존재하나, FFT 특징이 네트워크에 주입되지 않음 | 프라이어 생성 및 크로스 어텐션 추가 필요 |

## 3. 개발 단계 계획
| 단계 | 목표 | 상세 작업 | 산출물 |
| --- | --- | --- | --- |
| **S0 데이터 점검 (완료)** | 데이터 + 전처리 파이프라인 확보 | Drive/Colab 압축 해제 및 HF 저장 확인, repo에 노트북 추가 | `Data_Preparation.ipynb` |
| **S1 구조 프라이어 주입 (현재 단계)** | FFT 기반 저주파 프라이어 계산 후 U-Net 디코더에 크로스 어텐션 삽입 | `models/diffusion.py`에 구조 맵 생성, 레벨별 `FrequencyGuidedCrossAttention` 추가, config 플래그로 제어 | 코드 변경 + config 플래그 |
| **S2 학습 파이프라인 정비** | 실제 LDFDCT 데이터 학습 준비 | `configs/*` 데이터 경로 및 배치 사이즈 조정, `fast_ddpm_main.py`로 dummy/test run, TensorBoard 세팅 | 갱신된 config + sanity log |
| **S3 실험 & 평가** | Base 실험, Ablation(FFT × Cross-Attn) | 학습, 체크포인트 저장, `--sample --fid` 파이프, 구조 보존 비교 시각화 | 실험 기록, 샘플 이미지, FID |
| **S4 문서화** | 결과 공유 및 재현성 확보 | README/보고서 업데이트, 데이터 사용 가이드, 실험 노트 | 최종 보고 또는 wiki |

## 4. 즉시 실행 항목
1. `use_freq_cross_attn` 옵션을 추가하고, LearnableFourier 출력에서 저주파 구조 맵을 복원하여 크로스 어텐션으로 주입한다.  
2. 로컬 `data/LDFDCT/...` 구조를 확정하고 config의 `train/val/sample_dataroot` 값을 실제 경로로 교정한다.  
3. Dummy run(`python fast_ddpm_main.py --test --config configs/ldfd_linear.yml`)으로 파이프라인을 검증하고, 이후 전체 학습을 시작한다.

## 5. 리스크 및 대응
- **저주파 프라이어 품질**: LearnableFourier에서 사용한 주파수 서브샘플링 비율에 따라 구조 맵이 왜곡될 수 있으므로 `hard_thresholding_fraction` 튜닝과 `torch.tanh` 정규화를 활용.  
- **데이터 입출력**: 현재 HF datasets 포맷과 PNG pair 포맷이 혼재 가능 → 하나로 통일하거나 `fusion_datasets/LDFDCT.py`를 확장해 HF 저장소를 직접 읽도록 개선 필요.  
- **GPU 메모리**: 크로스 어텐션이 디코더 모든 해상도에 붙으면 메모리 상승 → config 플래그로 비활성화 가능하게 설계.  
- **실험 재현성**: 새 기능 적용 후 baseline 대비 개선 수치를 기록하고, `tensorboard`/`logs` 폴더에 config snapshot을 자동으로 남긴다.
