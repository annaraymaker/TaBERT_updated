# TaBERT_updated
TaBERT with updated dependencies.

# Make sure you have an unzipped tabert model to run the run.py file for testing!

```

cd fairseq-0.12.3
pip3 install -e .
cd ..

cd hydra-1.3.2
pip3 install -e .
cd ..

pip3 install --editable=git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install pytorch-scatter -c pyg

cd TaBERT
pip3 install -e .
./scripts/setup_env.sh
cd ..

# If this runs and prints some stuff, it works!
python3 run.py


```
