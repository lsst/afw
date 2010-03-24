import os, re
import eups

def ConfigureDependentProducts(productName, dependencyFilename="dependencies.dat"):
    """Process a product's dependency file, returning a list suitable for passing to SconsUtils.makeEnv"""
    productDir = eups.productDir(productName)
    if not productDir:
        raise RuntimeError, ("%s is not setup" % productName)

    dependencies = os.path.join(productDir, "etc", dependencyFilename)

    try:
        fd = open(dependencies)
    except:
        raise RuntimeError, ("Unable to lookup dependencies for %s" % productName)

    dependencies = []

    for line in fd.readlines():
        if re.search(r"^\s*#", line):
            continue

        mat = re.search(r"^(\S+):\s*$", line)
        if mat:
            dependencies += ConfigureDependentProducts(mat.group(1))
            continue
        #
        # Split the line into "" separated fields
        #
        line = re.sub(r"(^\s*|\s*,\s*|\s*$)", "", line) # remove whitespace and commas in the config file
        dependencies.append([f for f in re.split(r"['\"]", line) if f])

    return dependencies
    
    
