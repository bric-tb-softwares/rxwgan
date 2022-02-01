import os
basepath = os.getcwd()

path = basepath + '/wgangp'

# from...
exec_cmd = "(rm -rf rxwgan || true) && (rm -rf .complete || true) && " # some protections
exec_cmd+= "git clone https://github.com/bric-tb-softwares/rxwgan.git && "
# exec this
exec_cmd+= "cd rxwgan && export PYTHONPATH=$PYTHONPATH:$PWD/rxwgan/rxwgan && cd .. && "
exec_cmd+= "python rxwgan/share/job_tuning.py -j %IN -i %DATA -t 1 --test {TEST} -v %OUT "
exec_cmd+= "&& rm -rf rxwgan"

command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.Shenzhen.tuberculosis.model_wgangp.test_{TEST} \
  -c user.jodafons.brics.10sorts \
  -d user.jodafons.Shenzhen_table_from_raw.csv \
  --exec "{EXEC}" \
  --queue "gpu" \
  --bypass_local_test \
  """

try:
    os.makedirs(path)
except:
    pass

for test in range(5):
    cmd = command.format(PATH=path,EXEC=exec_cmd.format(TEST=test), TEST=test)
    print(cmd)
    os.system(cmd)


