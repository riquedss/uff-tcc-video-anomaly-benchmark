# Run INSIDE the container (see root Makefile: `make up-anomaly`).
# From there: cd AnomalyDetectionCVPR2018-Pytorch && make train ...

# Caminhos dentro do container: /data é o mount de DATA_ROOT (ver .env).
DATA_DIR ?= /data/ucf-crime
FEATURES_PATH ?= $(DATA_DIR)/features/$(MODEL_TYPE)
ANNOTATION_PATH ?= Train_Annotation.txt
TEST_ANNOTATION_PATH ?= Test_Annotation.txt
MODEL_PATH ?= exps/c3d/models/epoch_80000.pt
DATASET_PATH ?= $(DATA_DIR)/videos
# mfnet é o único peso já presente em pretrained/. Para c3d, baixe o
# c3d.pickle e passe MODEL_TYPE=c3d PRETRAINED_3D=pretrained/c3d.pickle.
MODEL_TYPE ?= mfnet
PRETRAINED_3D ?= pretrained/MFNet3D_UCF-101_Split-1_96.3.pth

.PHONY: extract-features train roc

extract-features:
	python feature_extractor.py \
		--dataset_path "$(DATASET_PATH)" \
		--model_type "$(MODEL_TYPE)" \
		--pretrained_3d "$(PRETRAINED_3D)" \
		--save_dir "$(FEATURES_PATH)"

train:
	python TrainingAnomalyDetector_public.py \
		--features_path "$(FEATURES_PATH)" \
		--annotation_path "$(ANNOTATION_PATH)"

roc:
	python generate_ROC.py \
		--features_path "$(FEATURES_PATH)" \
		--annotation_path "$(TEST_ANNOTATION_PATH)" \
		--model_path "$(MODEL_PATH)"
