#!/usr/bin/env python
"""
TestPolicy

Description
    Tests the Policy class
"""
from fw.Policy import Policy

import sys
import os

#==============================================================================
# Initialize the configuration dictionary
#=============================================================================
try:
    conf_file = os.environ['LSSTProto'] + '/tests/Policy/data/TestPolicy.conf'
except:
    conf_file = "./TestPolicy.conf"

print "-------------------------------------------------------------------"
print "Test Policy configuration filename: ",conf_file
policy = Policy(conf_file)
config_dict = policy.conf

print "-------------------------------------------------------------------"
print "Print the entire dictionary\n"
print config_dict

#=============================================================================
# Set policy parameters using defaults if the keyword is not found
#=============================================================================
print "\n-------------------------------------------------------------------"
print "Set policy parameters using defaults if the keyword is not found"
try:
    scaleratio = config_dict['scaleratio']
except:
    scaleratio = 0.23

try:
    ntri = config_dict['ntri']
except:
    ntri = 3

try:
    NotDefined = config_dict['NotDefined']
except:
    NotDefined = "Not defined in config_dict"

print ("scaleratio: %f   ntri: %d   NotDefined: %s" % ( scaleratio,ntri, NotDefined))
#=============================================================================
# If default is 0.0  the following simpler form works
#=============================================================================
#   NOTE:  use of 'eval' allows simple arithmetic ops in the <value> portion
#          Don't forget to use try/catch to find missing/bad data
print "\n-------------------------------------------------------------------"
try:
    Filter_MountTime = eval(str(config_dict["Filter_MountTime"])) 
    print ("Filter_MountTime: %f" % Filter_MountTime)
except:
    print "Failed in parsing: Filter_MountTime"

print "\nExpect failure in error processing to follow:"
try:
    NotDefined = eval(str(config_dict["NotDefined"])) 
    print ("NotDefined: %f" % NotDefined)
except:
    print "Expected failure in parsing: NotDefined"


#=============================================================================
# Demonstrate one form of array input.   
#=============================================================================
print "\n-------------------------------------------------------------------"
print "Demonstrate one form of array input, alternate error check used"
if ( config_dict.has_key ('userRegion')) :
    userRegion =  config_dict["userRegion"]
else :
    userRegion =  None
                                                                                
if (not isinstance(userRegion,list)):
    # turn it into a list with one entry
    save = userRegion
    userRegion = []
    userRegion.append(save)

for i in range (len(userRegion)):
    print ("UserRegion[%d]: " % (i)), userRegion[i]


#=============================================================================
# Demonstrate a form of psuedo array input.  
#=============================================================================
print "\n-------------------------------------------------------------------"
print "Demonstrate psuedo array format (4 arrays printed per row); no error check"
nobType = []
triggerAt = []
seqnStart = []
dumpIdle = []
i = 0
nobCount = config_dict["nobCount"]
print ("i: %d  nobCount: %d" % ( i,nobCount))

for i in range(int(nobCount)):
    nobType.append(config_dict["nobType[" + str(i) + "]"])
    triggerAt.append(config_dict['triggerAt[' + str(i) + '][]'])
    seqnStart.append(config_dict["seqnStart[" + str(i) + "]"])
    dumpIdle.append(config_dict["dumpIdle[" + str(i) + "]"])
    print "Row:",i," ", nobType[i]," ",triggerAt[i]," ",seqnStart[i]," ",dumpIdle[i]
    ++i

#=============================================================================
# Demonstrate use of conversion method to check input format
#=============================================================================
print "\n-------------------------------------------------------------------"
print "Demonstrate use of conversion method to check input format"
try:
    sbDateScale = float(config_dict['sbDateScale'])
except:
    sbDateScale = 3.0
#-------------------------------
print "sbDateScale: value entered as \'High\' but exception branch set it to: %f" % sbDateScale
