DATASET ?= ucf

.PHONY: train infer

train:
	python main.py --dataset '$(DATASET)' --mode 'train'

infer:
	python main.py --dataset '$(DATASET)' --mode 'infer'
