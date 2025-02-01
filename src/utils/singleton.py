class Singleton(type):
    """
    Implementation of metaclass based Singleton pattern for Python 3 taken from
    https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python
    This is a frequently used pattern but not covered in the Standard Library

    Usage in Python 3:
    class MyClass(BaseClass, metaclass=Singleton):
        pass

    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
