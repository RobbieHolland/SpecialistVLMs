def subclasses(cls):
    return [getattr(cls, attr) for attr in cls.__dict__ if isinstance(getattr(cls, attr), type)]