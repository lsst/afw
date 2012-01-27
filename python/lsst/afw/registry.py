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

class RegistryAdaptor(object):
    """Private class that makes a Registry behave like the thing a ConfigChoiceField expects."""

    def __init__(self, registry):
        self.registry = registry

    def __getitem__(self, k):
        return self.registry[k].ConfigClass

class FactoryConfigInstanceDict(lsst.pex.config.ConfigInstanceDict):

    def __init__(self, fullname, typemap, multi, history=None):
        lsst.pex.config.ConfigInstanceDict.__init__(self, fullname, typemap, multi, history)

    def _getFactory(self):
        if self._multi:
            raise AttributeError("Multi-selection field %s has no attribute 'factory'" % self._fullname)
        return self.typemap.registry[self._selection].factory
    factory = property(_getFactory)

    def _getFactories(self):
        if not self._multi:
            raise AttributeError("Single-selection field %s has no attribute 'factories'" % self._fullname)
        return [self.typemap.registry[c].factory for c in self._selection]
    factories = property(_getFactories)

    def apply(self, *args, **kwds):
        """Call the active factory with the active config as the first argument.

        If this is a multi-selection field, return a list obtained by calling each active
        factories with its corresponding active config as its first argument.

        Additional arguments will be passed on to the factory or factories.
        """
        if self._multi:
            return [self.typemap.registry[c].factory(self[c], *args, **kwds) for c in self._selection]
        else:
            return self.factory(self.active, *args, **kwds)

class RegistryField(lsst.pex.config.ConfigChoiceField):

    def __init__(self, doc, registry, default=None, optional=False, multi=False,
                 instanceDictClass=FactoryConfigInstanceDict):
        typemap = RegistryAdaptor(registry)
        lsst.pex.config.ConfigChoiceField.__init__(self, doc, typemap, default, optional, multi,
                                                   instanceDictClass)

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
