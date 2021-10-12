
__all__ = ['declare_property']

def declare_property( cls, kw, name, value , private=False):
  atribute = ('__' + name ) if private else name
  if name in kw.keys():
    setattr(cls,atribute, kw[name])
  else:
    setattr(cls,atribute, value)


