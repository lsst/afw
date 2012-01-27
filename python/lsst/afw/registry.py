import lsst.pex.config

__all__ = ("Registry", "registerConfig", "registerFactory")

class Registry(object):
    """A base class for global registries, mapping names to config/factory pairs.

    There are no requirements on what the factory object produces - it is just expected
    to be a callable object whose first argument is a Config instance.
    """

    def __init__(self):
        self._dict = {}

    def register(self, name, config, factory):
        """Add a new item to the registry.

        @param factory    A callable who takes a Config instance as its first argument.  A factory may
                          be a Python type, but is not required to be.
        @param config     A Config class used to configure the factory.
        """
        if config is None:
            config = factory.ConfigClass
        self._dict[name] = (config, factory)

    def getConfig(self, name):
        """Return a the Config class registered with the given name.
        """
        config, factory = self._dict[name]
        return config

    def getFactory(self, name):
        """Return the factory (not copied) that was registered with the given name.
        """
        config, factory = self._dict[name]
        return factory

    def getNames(self):
        """Return the list of registered names.
        """
        return self._dict.keys()

def makeRegistry(doc):
    """A convenience function to create a new registry.
    
    The returned value is an instance of a trivial subclass of Registry whose only purpose is to
    customize its docstring.
    """
    cls = type("Registry", (Registry,), {"__doc__": doc})
    return cls()

def registerFactory(name, registry, config=None):
    """A decorator that adds a class as a factory in a Registry.

    If the 'config' argument is None, the class's ConfigClass attribute will be used.
    """
    def decorate(cls):
        if config is None:
            config = cls.ConfigClass
        registry.register(name, factory=cls, config=config)
        return cls
    return decorate

def registerConfig(name, registry, factory):
    """A decorator that adds a class as a config in a Registry, and associates it with the given factory.
    """
    def decorate(cls):
        registry.register(name, factory=factory, config=cls)
        return cls
    return decorate
