import os
basepath = os.getcwd()

path = basepath

#
# Prepare my job script!
#

# remove all by hand in case of job retry... NOTE: some jobs needs to recover s
exec_cmd = "(rm -rf rxcore || true) && " # some protections
exec_cmd = "(rm -rf rxwgan || true) && " # some protections

# download all necessary local packages...
exec_cmd+= "git clone https://github.com/bric-tb-softwares/rxcore.git && "
exec_cmd+= "git clone https://github.com/bric-tb-softwares/rxwgan.git && "
# setup into the python path
exec_cmd+= "cd rxcore && export PYTHONPATH=$PYTHONPATH:$PWD/rxcore && cd .. && "
exec_cmd+= "cd rxwgan && export PYTHONPATH=$PYTHONPATH:$PWD/rxwgan && cd .. && "

# execute my job!
exec_cmd+= "python rxwgan/versions/v1/v1_tb/job_tuning.py -j %IN -i %DATA -v %OUT && "

# if complete, remove some dirs...
exec_cmd+= "rm -rf rxwgan && rm -rf rxcore"

command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.task.Shenzhen.model_wgangp.v1_tb.test_{TEST} \
  -c user.jodafons.job.Shenzhen.model_wgangp.10sorts.test_{TEST} \
  -d user.jodafons.data.Shenzhen_table_from_raw.csv \
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


