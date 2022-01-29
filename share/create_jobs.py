
import json

output = 'job.wgangp.sort_%d.json'

for sort in range(10):

        d = {   
                'sort': sort,
            }

        o = output%( sort)
        with open(o, 'w') as f:
            json.dump(d, f)





