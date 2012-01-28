import lsst.pex.config as pexConfig
import collections

__all__ = ("Registry", "makeRegistry", "registerConfig", "registerFactory")

class FactoryWrapper(object):
    """A wrapper for factories
    
    Used for factories that don't contain a ConfigClass attribute,
    or contain one that is being overridden.
    """
    def __init__(self, factory, ConfigClass):
        self.ConfigClass = ConfigClass
        self._factory = factory

    def __call__(self, *args, **kwargs):
        return self._factory(*args, **kwargs)
    

class Registry(collections.Mapping):
    """A base class for global registries, mapping names to factories.

    There are no hard requirements on factory, but they typically create an algorithm
    or are themselves the algorithm, and typical usage is as follows:
    - factory is a callable whose call signature is (config, ...extra arguments...)
    - All factories added to a particular registry will have the same call signature
    - All factories in a registry will typical share something important in common.
      For example all factories in psfMatchingRegistry return a psf matching
      class that has a psfMatch method with a particular call signature.

    A registry acts like a read-only dictionary with an additional register method to add items.
    The dict contains factories and each factory has an instance ConfigClass.

    Example:
    registry = Registry()
    class FooConfig(pexConfig.Config):
        val = pexConfig.Field(dtype=int, default=3, doc="parameter for Foo")
    class Foo(object):
        ConfigClass = FooConfig
        def __init__(self, config):
            self.config = config
        def addVal(self, num):
            return self.config.val + num
    registry.register("foo", Foo)
    names = registry.keys() # returns ("foo",)
    fooFactory = registry["foo"]
    fooConfig = fooItem.ConfigClass()
    foo = fooFactory(fooConfig)
    foo.addVal(5) # returns config.val + 5
    """

    def __init__(self, configBaseType=pexConfig.Config):
        """Construct a registry of name: factories
        
        @param configBaseType: base class for config classes in registry
        """
        if not issubclass(configBaseType, pexConfig.Config):
            raise TypeError("configBaseType=%r must be a subclass of pexConfig.Config" % (configBaseType,))
        self._configBaseType = configBaseType
        self._dict = {}

    def register(self, name, factory, ConfigClass=None):
        """Add a new item to the registry.
        
        @param factory    A callable that takes a Config instance as its first argument.  A factory may
                          be a Python type, but is not required to be.
        @param ConfigClass  A subclass of pex_config Config used to configure the factory;
                          if None then factory.ConfigClass is used.
                          
        @note: if ConfigClass is provided then the supplied factory is wrapped in a new object.
        Otherwise the original factory is stored.
        
        @raise AttributeError if ConfigClass is None and factory does not have attribute ConfigClass
        """
        if name in self._dict:
            raise RuntimeError("An item with name %r already exists" % (name,))
        if ConfigClass is None:
           wrapper = factory
        else:
            wrapper = FactoryWrapper(factory, ConfigClass)
        if not issubclass(wrapper.ConfigClass, self._configBaseType):
            raise TypeError("ConfigClass=%r is not a subclass of %r" % (self._configBaseType,))
        self._dict[name] = wrapper
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __len__(self):
        return len(self._dict)
    
    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def makeField(self, doc, default=None, optional=False, multi=False):
        return RegistryField(doc, self, default, optional, multi)

class RegistryAdaptor(object):
    """Private class that makes a Registry behave like the thing a ConfigChoiceField expects."""

    def __init__(self, registry):
        self.registry = registry

    def __getitem__(self, k):
        return self.registry[k].ConfigClass

class FactoryConfigInstanceDict(pexConfig.ConfigInstanceDict):

    def __init__(self, fullname, typemap, multi, history=None):
        pexConfig.ConfigInstanceDict.__init__(self, fullname, typemap, multi, history)

    def _getFactory(self):
        if self._multi:
            raise AttributeError("Multi-selection field %s has no attribute 'factory'" % self._fullname)
        return self.types.registry[self._selection]
    factory = property(_getFactory)

    def _getFactories(self):
        if not self._multi:
            raise AttributeError("Single-selection field %s has no attribute 'factories'" % self._fullname)
        return [self.types.registry[c] for c in self._selection]
    factories = property(_getFactories)

    def applyFactory(self, *args, **kwds):
        """Call the active factory with the active config as the first argument.

        If this is a multi-selection field, return a list obtained by calling each active
        factory with its corresponding active config as its first argument.

        Additional arguments will be passed on to the factory or factories.
        """
        if self._multi:
            return [self.types.registry[c](self[c], *args, **kwds) for c in self._selection]
        else:
            return self(self.active, *args, **kwds)

class RegistryField(pexConfig.ConfigChoiceField):

    def __init__(self, doc, registry, default=None, optional=False, multi=False,
                 instanceDictClass=FactoryConfigInstanceDict):
        types = RegistryAdaptor(registry)
        pexConfig.ConfigChoiceField.__init__(self, doc, types, default, optional, multi, instanceDictClass)

def makeRegistry(doc, configBaseType=pexConfig.Config):
    """A convenience function to create a new registry.
    
    The returned value is an instance of a trivial subclass of Registry whose only purpose is to
    customize its doc string and set attrList.
    """
    cls = type("Registry", (Registry,), {"__doc__": doc})
    return cls(configBaseType=configBaseType)

def registerFactory(name, registry, ConfigClass=None):
    """A decorator that adds a class as a factory in a Registry.

    If the 'ConfigClass' argument is None, the class's ConfigClass attribute will be used.
    """
    def decorate(cls):
        registry.register(name, factory=cls, ConfigClass=ConfigClass)
        return cls
    return decorate

def registerConfig(name, registry, factory):
    """A decorator that adds a class as a ConfigClass in a Registry, and associates it with the given factory.
    """
    def decorate(cls):
        registry.register(name, factory=factory, ConfigClass=cls)
        return cls
    return decorate
