"""
Policy provides the method of delivering default and/or user specified
parameters to a class in order to direct the class's operation.
"""

import os

#****************************************************************************
class Policy(object):
    #------------------------------------------------------------------------
    def __init__ (self, fileName=None, kws=None, defDict=None):
        """
        Parse the configuration file (fileName) and return a dictionary of
        the content of the file.
        
        fileName must be an ASCII file containing a list of key = value
        pairs, one pair per line. Comments are identified by a '#' and can
        be anywhere in the file. Everything following a '#' (up to the 
        carriage return/new line) is considered to be a comment and 
        ignored.
        
        The dictionary has the form {key: value} where value is a simple 
        number if and only if key appears only one time in fileName. 
        Otherwise, value is an array.
        
        Value can have '=' sign in it: each non-comment line is split 
        using the '=' character as delimiter, only once and starting from 
        the left. Extra white spaces and '\n' characters are stripped from
        both key and value.
        
        An attempt is made to convert value into a float. If that fails, 
        value is assumed to be a string.
        
        
        Input
        fileName:   the name of the configuration file. If the provided name
                    is not rooted at '/', the environment variable 
                    LSST_POLICY_DIR will be prepended to create a full pathname
        kws:        a dict of override values
        defDict:    a dict of default values

        Required environment variable
        LSST_POLICY_DIR:        pathname to directory containing all policy
                                statements needed by currently active code
        
        Return
        A Policy object
        
        Raise
        IOError if fileName cannot be opened for reading.
        """

        self.conf = {}

        policyDir = os.getenv("LSST_POLICY_DIR")
        if policyDir == None:
             raise IOError("environment variable LSST_POLICY_DIR not defined")
        if fileName[0:1] != '/':
            fileName = os.path.join(policyDir,fileName)

        if fileName:
            # Try to read the file fileName (raise IOError if something bad 
            # happens).
            policyFile = file (fileName, 'rU')
            try:
                for line in policyFile:
                    # Skip blank lines and lines that start with "#"
                    line = line.strip()
                    if not line or line[0]=='#': 
                        continue

                    # Remove end comment
                    good = line.split ('#', 1)[0]
    
                    # Split on the first occurrence of the '=' sign
                    words = good.split ('=', 1)
    
                    if len (words) != 2:
                        continue
                        
                    key = words[0].strip()
                    val = words[1].strip()

                    # Try and convert val to a float
                    try:
                        val = float (val)
                    except (ValueError, TypeError):
                        # Oops, must be a string, then
                        pass

                    if not self.conf.has_key (key):
                        self.conf[key] = val
                    else:
                        v = self.conf[key]

                        # Do we have an array, already?
                        if (isinstance (v, list)):
                            self.conf[key].append (val)
                        else:
                            self.conf[key] = [v, val]
            finally:
                policyFile.close()

        # Set some class attributes if they exist
        self.clientPolicyVersion = self.get("clientPolicyVersion", "")
        self.serverPolicyVersion = self.get("serverPolicyVersion", "")
        self.clientPolicyName = self.get("clientPolicyName", "")
        
        # Handle defaults
        if defDict:
            for key, val in defDict.iteritems():
                self.conf.setdefault(key, val)

        # If any kw-val pairs were supplied in the optional kws dictionary, update.
        # If a keyword is already present in self.conf, override its value with the
        # one supplied.   If it is not, add the kw/val pair to self.conf.
        if (kws):
            self.conf.update(kws)
    
    #------------------------------------------------------------------------
    def getClientPolicyVersion (self):
        """
        Return version of active Policy configuration
        """
        return self.clientPolicyVersion 

    #------------------------------------------------------------------------
    def getServerPolicyVersion (self):
        """
        Return version of required Policy class
        """
        return self.serverPolicyVersion 

    #------------------------------------------------------------------------
    def getPolicyName (self):
        """
        Return the type of Policy
        """
        return self.clientPolicyName 

    #------------------------------------------------------------------------
    def get (self, kwName, defValue=None):
        """
        Return the value associated with the specified keyword.

        Input
            kwName      name of keyword; Format: string
            defValue    default value for keyword
        Return
            kwValue     value of keyword, if it exists, else defValue
        """
        return self.conf.get(kwName, defValue)
       
    Get = get # for backwards compatibility

    #------------------------------------------------------------------------
    def __getitem__(self, kwName):
        """
        Implement policy[kwName]
        
        Return the value associated with the specified keyword.
        Raise KeyError if kwName not found.
        """
        return self.conf[kwName]
