import os
basepath = os.getcwd()

path = basepath + '/wgangp/v1'

# from...
exec_cmd = "git clone https://github.com/bric-tb-softwares/rxwgan.git && "
# exec this
exec_cmd+= "cd rxwgan && export PYTHONPATH=$PYTHONPATH:$PWD/rxwgan/rxwgan && cd .. && "
exec_cmd+= "python rxwgan/share/run_wgangp.py -j %IN -i %DATA -t 1 --test {TEST} -v %OUT"

command = """maestro.py task create \
  -v {PATH} \
  -t user.jodafons.Shenzhen_wgangp.v1.tb.test_{TEST} \
  -c user.jodafons.bric.10sorts \
  -d user.jodafons.Shenzhen_table_from_raw.csv \
  --exec "{EXEC}" \
  --queue "gpu" \
  """

try:
    os.makedirs(path)
except:
    pass

for test in range(10):
    cmd = command.format(PATH=path,EXEC=exec_cmd.format(TEST=test), TEST=test)
    print(cmd)
    os.system(cmd)
    break


