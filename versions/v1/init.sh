
rm -rf rxcore || true
rm -rf rxwgan || true
rm -rf wandb  || true

export WANDB_API_KEY=$WANDB_API_KEY

git clone https://github.com/bric-tb-softwares/rxcore.git
git clone https://github.com/bric-tb-softwares/rxwgan.git

# setup into the python path
export PYTHONPATH=$PYTHONPATH:$PWD/rxcore
export PYTHONPATH=$PYTHONPATH:$PWD/rxwgan

echo $PYTHONPATH
ls -lisah