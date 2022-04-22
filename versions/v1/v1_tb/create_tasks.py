import os
basepath = os.getcwd()

path = basepath

datapath = '/home/jodafons/public/brics_data/Shenzhen/raw/Shenzhen_table_from_raw.csv'

exec_cmd = "source {PATH}/init.sh && "
exec_cmd+= "python {PATH}/run.py -j %IN -i {DATA} && "
exec_cmd+= "source {PATH}/end.sh"

exec_cmd = exec_cmd.format(PATH=basepath, DATA=datapath)

command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.task.Shenzhen.wgangp.v1_tb \
  -c jobs \
  --exec "{EXEC}" \
  """



command = command.format(PATH=path,EXEC=exec_cmd.format(TEST=test), TEST=test)
    print(cmd)
    os.system(cmd)
    break


