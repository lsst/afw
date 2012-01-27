import lsst.pex.config as pexConfig
import collections

__all__ = ("Registry", "makeRegistry", "registerConfig", "registerFactory")

class FactoryConfigPair(object):
    """A simple structure with two fields: factory and ConfigClass
    """
    def __init__(self, factory, ConfigClass):
        self.factory = factory
        self.ConfigClass = ConfigClass

class Registry(collections.Mapping):
    """A base class for global registries, mapping names to factory/ConfigClass pairs.

    There are no hard requirements on factory, but typical usage is as follows:
    - factory is a callable whose call signature is (config, ...extra arguments...)
    - All factories added to a particular registry will have the same call signature
    - All factories in a registry will typical share something important in common.
      For instance if factory is an algorithm registry (a common case), then the
      API of the returned algorithm should be the same for all items in the registry.

    A registry acts like a read-only dictionary with an additional register method to add items.
    The dict contains instances of FactoryConfigPair: an object with two attributes: factory and ConfigClass

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
    fooItem = registry["foo"]
    foo = fooItem.factory(fooItem.ConfigClass())
    foo.addVal(5) # returns config.val + 5
    """
    def __init__(self, configBaseType=pexConfig.Config):
        """Construct a registry of name: FactoryConfigPair objects
        
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
        
        @raise AttributeError if ConfigClass is None and factory does not have attribute ConfigClass
        """
        if name in self._dict:
            raise RuntimeError("An item with name %r already exists" % (name,))
        if ConfigClass is None:
            ConfigClass = factory.ConfigClass
        if not issubclass(ConfigClass, pexConfig.Config):
            raise TypeError("ConfigClass=%r is not a subclass of %r" % (self._configBaseType,))
        self._dict[name] = FactoryConfigPair(factory=factory, ConfigClass=ConfigClass)
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __len__(self):
        return len(self._dict)
    
    def __iter__(self):
        return iter(self._dict)

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
