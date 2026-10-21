# Official SwinIR source

Unmodified network_swinir.py from JingyunLiang/SwinIR commit 6545850fbf8df298df73d81f3e8cba638787c8bd, Apache-2.0; license included. Original authors: Jingyun Liang and contributors, based on Ze Liu and Swin Transformer/KAIR. Source: https://github.com/JingyunLiang/SwinIR .

Optional baseline dependencies: timm==1.0.30, torchvision==0.29.1+cpu (tested with torch==2.14.1+cpu). Official lightweight x2 weights are fetched separately using configs/official-swinir.json; never substitute random weights. This RGB natural-image baseline is not a multispectral model, and its bicubic pretraining differs from the laboratory degradation.
