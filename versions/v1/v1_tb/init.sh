
rm -rf rxcore || true
rm -rf rxwgan || true
rm -rf wandb  || true

export WANDB_API_KEY=$WANDB_API_KEY

git clone https://github.com/bric-tb-softwares/rxcore.git
git clone https://github.com/bric-tb-softwares/rxwgan.git

# setup into the python path
cd rxcore && export PYTHONPATH=$PYTHONPATH:$PWD/rxcore && cd ..
cd rxwgan && export PYTHONPATH=$PYTHONPATH:$PWD/rxwgan && cd ..
