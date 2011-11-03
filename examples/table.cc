#include <iostream>
#include <iterator>
#include <algorithm>

#include "lsst/catalog/Layout.h"

using namespace lsst::catalog;

int main() {

    LayoutBuilder builder;
    builder.add(Field< int >("myIntField", "an integer scalar field."));
    builder.add(Field< Array<double> >(5, "myDArrayField", "a double array field.", NOT_NULL));
    builder.add(Field< float >("myFloatField", "a float scalar field.", NOT_NULL));
    Layout layout = builder.finish();

    Layout::Description description = layout.describe();

    std::ostream_iterator<FieldDescription> osi(std::cout, "\n");
    std::copy(description.begin(), description.end(), osi);
    
}
