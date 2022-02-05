import os
basepath = os.getcwd()

path = basepath

#
# Prepare my job script!
#

# remove all by hand in case of job retry... NOTE: some jobs needs to recover s
exec_cmd = "(rm -rf rxcore || true) && " # some protections
exec_cmd+= "(rm -rf rxwgan || true) && " # some protections
exec_cmd+= "(rm -rf wandb || true) && " # some protections

exec_cmd+= "export WANDB_API_KEY=$WANDB_API_KEY && "
#exec_cmd+= "wandb login --relogin"
# download all necessary local packages...
exec_cmd+= "git clone https://github.com/bric-tb-softwares/rxcore.git && "
exec_cmd+= "git clone https://github.com/bric-tb-softwares/rxwgan.git && "
# setup into the python path
exec_cmd+= "cd rxcore && export PYTHONPATH=$PYTHONPATH:$PWD/rxcore && cd .. && "
exec_cmd+= "cd rxwgan && export PYTHONPATH=$PYTHONPATH:$PWD/rxwgan && cd .. && "

# execute my job!
exec_cmd+= "python rxwgan/versions/v1/v1_notb/job_tuning.py -j %IN -i %DATA -v %OUT && "

# if complete, remove some dirs...
exec_cmd+= "rm -rf rxwgan && rm -rf rxcore && "
exec_cmd+= "(rm -rf wandb || true)" # some protections

command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.task.Shenzhen.wgangp.v1_notb.test_{TEST} \
  -c user.jodafons.job.Shenzhen.wgangp.v1.test_{TEST}.10_sorts \
  -d user.jodafons.Shenzhen_table_from_raw.csv \
  --exec "{EXEC}" \
  --queue "gpu" \
  """

try:
    os.makedirs(path)
except:
    pass

tests = [0]

for test in tests:
    cmd = command.format(PATH=path,EXEC=exec_cmd.format(TEST=test), TEST=test)
    print(cmd)
    os.system(cmd)
    break


