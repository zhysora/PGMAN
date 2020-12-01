############## arguments ##############
models = ['cycle_psgan']  # model name
satellites = ['GF-2']       # dataset name
refs = [0, 1]        # ref or no-ref metrics
save = 1             # save or not
epochs = [20]

# config of dataset, you can find it in 'record.txt' under the specific dataset fold
# row and col is to joint patches images to origin size
test_cnt = { 0: {"QB": 828, "GF-2": 286, 'WV-3': 308}, 
             1: {"QB": 77, "GF-2": 286, 'WV-3': 308} }
row = { 0: {"QB": 23, "GF-2": 13, 'WV-3': 28},
        1: {"QB": 7, "GF-2": 13, 'WV-3': 28} } 
col = { 0: {"QB": 36, "GF-2": 22, 'WV-3': 11}, 
        1: {"QB": 11, "GF-2": 22, 'WV-3': 11} } 
bit = {"QB": 11, "GF-2": 10, "WV-3": 11}

############## shell ##############       
import os

for model in models:
    for satellite in satellites:
        for epoch in epochs:
            for ref in refs:
                dataset = satellite
                # for no-ref test we use 'origin dataset'
                if ref == 0: 
                    dataset_path = '/data/zhouhuanyu/PSData3/Dataset/{}/test_full_res'.format(dataset)
                    model_outpath = '/data/zhouhuanyu/PSData3/model_out/{}/out/test_{}_full_res_{}'.format(model.lower(), dataset, epoch)
                    save_path = '/data/zhouhuanyu/PSData3/Result/{}_full_res'.format(dataset)
                else:
                    dataset_path = '/data/zhouhuanyu/PSData3/Dataset/{}/test_low_res'.format(dataset)
                    model_outpath = '/data/zhouhuanyu/PSData3/model_out/{}/out/test_{}_low_res_{}'.format(model.lower(), dataset, epoch)
                    save_path = '/data/zhouhuanyu/PSData3/Result/{}_low_res'.format(dataset)
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                os.system('cd eval; nohup python -u eval_one.py \
                    --dataset_path %s\
                    --model %s\
                    --model_outpath %s\
                    --save_path %s\
                    --num %d \
                    --row %d \
                    --col %d \
                    --ref %d \
                    --save %d \
                    --bit %d \
                    > ../model_out/%s/log/eval_%s-%d.log_%d 2>&1 & ' % (
                    dataset_path,
                    model.lower(),
                    model_outpath,
                    save_path,
                    test_cnt[ref][dataset], 
                    row[ref][dataset], 
                    col[ref][dataset], 
                    ref, 
                    save,
                    bit[dataset],
                    model.lower(), dataset, ref, epoch
                    ) 
                )
