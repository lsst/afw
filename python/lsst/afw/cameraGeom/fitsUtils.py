import re, warnings

def getByKey(metadata, key):
    """Wrapper for getting a value from a metadata object by key.
       @param[in] metadata  metadata object to query for value
       @param[in] key       key to use for value lookup
       @returns value associated with key, None if key does not exist
    """
    mdKeys = metadata.paramNames()
    if key in mdKeys:
        return metadata.get(key)
    else:
        return None

def setByKey(metadata, key, value, clobber):
    """Wrapper for setting a value in a metadata object.  Deals with case
       where the key already exists.
       @param[in, out] metadata  metadata object ot modify in place.
       @param[in] key       key to associate with value
       @param[in] value     value to assign in the metadata object
       @param[in] clobber   Clobber the value if the key already exisists?
    """
    mdKeys = metadata.paramNames()
    if key not in mdKeys or (key in mdKeys and clobber):
        metadata.set(key, value)

class HeaderMap(dict):
    """ Class to hold mapping of header cards to attributes"""
    def addEntry(self, keyname, attribute_name, default=None, transform=lambda x: x):
        """Adds an entry to the registr
           @param[in] keyname         Key used to retrieve the header record
           @param[in] attribute_name  Name of the attribute to store the value in
           @param[jn] default         Default velue to store if the header card is not available
           @param[in] transform       Transform to apply to the header value before assigning it to the
                                      attribute.
        """
        self.__setitem__(attribute_name, {'keyName':keyname,
                                         'default':default,
                                         'transform':transform})

    def setAttributes(self, obj, metadata, doRaise=True):
        """Sets the attributes on the give object given a metadata object.
           @param[in, out] obj       Object on which to operate in place
           @param[in]      metadata  Metadata object used for applying the mapping
           @param[in]      doRaise   Raise exceptions on calling methods on the input object that do not exist?
        """
        for key, attrDict in self.iteritems():
            try:
                value = getByKey(metadata, attrDict['keyName'])
                if value is not None:
                    self._applyVal(obj, value, key, attrDict['transform'])
                else:
                    #Only apply transform if the metadata has a value for this key
                    #otherwise assume the default value is transformed.
                    value = attrDict['default']
                    self._applyVal(obj, value, key, lambda x: x)
            except Exception, e:
                if doRaise:
                    raise
                else:
                    warnings.warn('WARNING: Failed to set %s attribute with %s value: %s'%
                                  (key, value, e.message))

    def _applyVal(self, obj, value, attrName, transform):
        raise NotImplementedError('Must be implemented in sub-class')

class HeaderAmpMap(HeaderMap):
    """ Class to hold mapping of header cards to AmpInfoTable attributes
        The amp info is stored using setters, thus calling the attribute as a function.
    """
    def _applyVal(self, obj, value, attrName, transform):
        getattr(obj, attrName)(transform(value))

class HeaderDetectorMap(HeaderMap):
    """ Class to hold mapping of header cards to Detector attributes
        Detector information is stored as attributes on a Config object.
    """
    def _applyVal(self, obj, value, attrName, transform):
        obj.__setattr__(attrName, transform(value))
