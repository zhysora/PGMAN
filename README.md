# PGMAN

code for "PGMAN: An Unsupervised Generative Multi-adversarial Network for Pan-sharpening"

implemented in PyTorch1.1

python lib see in requirement.txt

# Raw Data

link：https://pan.baidu.com/s/190MywbwIlvONA_9-6-KMtQ 

code：u041 

# Quick Start

first you should download the raw data, and then build the dataset. 
```
python data/handle_raw.py
python data/gen_dataset.py
```
You may need to modify the corresponding path in the code before excuting it.

the main pipeline is in the 'main.py', for quick start, you can just run the 'run.py'. 
```
python run.py
```
You may need to modify the corresponding parameters in the 'run.py' before excuting it.
