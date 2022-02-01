
import json

#output = 'job.Shenzhen.tuberculosis.sort_%d.json'
output = 'job.Shenzhen.notuberculosis.sort_%d.json'

for sort in range(10):

        d = {   
                'sort'   : sort,
                'target' : 0, # or 0
            }

        o = output%( sort)
        with open(o, 'w') as f:
            json.dump(d, f)





