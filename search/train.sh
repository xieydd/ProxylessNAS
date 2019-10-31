python -m torch.distributed.launch -nproc_prc_node=1 train_search_apex.py --name=proxylessnas --dataset=imagenet --gpus=0 --batch_size=256 --target_hardware=mobile --warmup_epochs=40 --epochs=150 --workers=30 --train_portion=1.0 --val_portion=0.01