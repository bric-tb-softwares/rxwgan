import os
basepath = os.getcwd()

path = basepath



exec_cmd = "source {PATH}/init.sh && "
exec_cmd+= "python {PATH}/job_tuning.py -j %IN -i %DATA -v %OUT && "
exec_cmd+= "source {PATH}/end.sh"



command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.task.Shenzhen.wgangp.v1_tb.test_{TEST} \
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


