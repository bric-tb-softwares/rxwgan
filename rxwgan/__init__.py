__all__ = ['declare_property']

#
# declare property
#
def declare_property( cls, kw, name, value , private=False):
  atribute = ('__' + name ) if private else name
  if name in kw.keys():
    setattr(cls,atribute, kw[name])
  else:
    setattr(cls,atribute, value)


from . import wgangp
__all__.extend( wgangp.__all__ )
from .wgangp import *

from . import plots
__all__.extend( plots.__all__ )
from .plots import *

from . import stats
__all__.extend( stats.__all__ )
from .stats import *

from . import metrics
__all__.extend( metrics.__all__ )
from .metrics import *

from . import stratified_kfold
__all__.extend( stratified_kfold.__all__ )
from .stratified_kfold import *