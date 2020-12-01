############## arguments ##############
models = ['PGMAN'] #['pnn', 'msdcnn', 'drpnn', 'pannet', 'tfnet', 'psgan']    # model name
datasets = ["QB"]       # dateset name
gpuset = [5]           # gpu ids
batch_size = 8         # batch-size
cuda = True            # use cuda
onlyTest = False        # Test mode

if cuda:
    cuda = "--cuda"
else:
    cuda = ""

############## shell ##############
import os
indx = 0
for model in models:
    if not os.path.exists('model_out/%s' % model):
        os.makedirs('model_out/%s' % model)
    if not os.path.exists('model_out/%s/log' % model):
        os.makedirs('model_out/%s/log' % model)
            
    for dataset in datasets:
        gpus = gpuset[indx]
        print ("run model:{} dataset:{} on gpu {}".format(model, dataset, gpus))
        indx = indx + 1
        if not onlyTest: # train & test
            os.system('nohup python -u main.py \
                %s \
                --gpus %d\
                --model %s\
                --dataset /data/zhouhuanyu/PSData3/Dataset/%s\
                --outpath /data/zhouhuanyu/PSData3\
                --batch_size %d\
                --epochs 20\
                --save_freq 20\
                --test_freq 20\
                > model_out/%s/log/nohup_%s.log 2>&1 &' % 
                (cuda, gpus, model, dataset, batch_size, model, dataset)
            )
        else: # test
            os.system('nohup python -u main.py \
                %s \
                --gpus %d\
                --model %s\
                --dataset /data/zhouhuanyu/PSData3/Dataset/%s\
                --outpath /data/zhouhuanyu/PSData3 \
                --batch_size %d\
                --epochs 20\
                --save_freq 20\
                --test_freq 20\
                --checkpoint /data/zhouhuanyu/PSData3/model_out/%s/out/train_%s/model_epoch_20.pth\
                > model_out/%s/log/nohup_%s.log 2>&1 &' % 
                (cuda, gpus, model, dataset, batch_size, model, dataset, model, dataset)
            )
