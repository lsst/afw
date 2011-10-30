#include <iostream>
#include <iterator>
#include <algorithm>

#include "lsst/catalog/Layout.h"

using namespace lsst::catalog;

int main() {

    Layout layout;
    layout.addField(Field<int>("myIntField", "an integer scalar field."));
    layout.addField(Field< Vector<double> >("myDVecField", "a double vector field.", 5));
    layout.addField(Field< float >("myFloatField", "a float scalar field."));

    Layout::Description description = layout.describe();

    std::ostream_iterator<FieldDescription> osi(std::cout, "\n");
    std::copy(description.begin(), description.end(), osi);
    
}
