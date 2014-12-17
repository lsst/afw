class CameraGeomBuilderConfig(pexConfig.Config):
    raise NotImplementedError()

class HeaderMap(dict):
    """ Class to hold mapping of header cards to attributes"""
    def __init__(self, keyname, attribute_name, default, transform=None):
        self.keyname = keyname
        self.attribute_name = attribute_name
        self.transform = transform
        self.default = default

    def addEntry(self, keyname, attribute_name, default=None, transform=None):
        self.__setitem__(self, keyname, {'attrName':attribute_name,
                                         'default':default,
                                         'transform':transform})

    def setAttributes(self, obj, metadata):
        for key, attrDict in self.iteritems():
            value = metadata.get(key)
            if value is not None:
                self._applyVal(obj, value, attrDict['attrName'], attrDict['transform'])
            else:
                #Only apply transform if the metadata has a value for this key
                #otherwise assume the default value is transformed.
                self._applyVal(obj, value, attrDict['attrName'], None)

    def _applyVal(self, obj, value, attrName, transform):
        raise NotImplementedError('Must be implemented in sub-class')

class HeaderAmpMap(HeaderMap):
    """ Class to hold mapping of header cards to AmpInfoTable attributes"""
    def _applyVal(self, obj, value, attrName, transform):
        obj.get_attribute(attrName)(transform(value))

class HeaderDetectorMap(HeaderMap):
    """ Class to hold mapping of header cards to Detector attributes"""
    def _applyVal(self, obj, value, attrName, transform):
        obj.__setattr__(attrName, transform(value))

