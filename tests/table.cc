#include <iostream>
#include <iterator>
#include <algorithm>

#include "lsst/catalog/Layout.h"
#include "lsst/catalog/TableBase.h"

using namespace lsst::catalog;

int main() {

    LayoutBuilder builder;
    
    Key<int> myIntField = builder.add(Field< int >("myIntField", "an integer scalar field."));
    
    Key< Array<double> > myArrayField 
        = builder.add(Field< Array<double> >(5, "myArrayField", "a double array field.", NOT_NULL));
    
    builder.add(Field< float >("myFloatField", "a float scalar field.", NOT_NULL));

    Layout layout = builder.finish();

    Key<float> myFloatField = layout.find<float>("myFloatField");

    Layout::Description description = layout.describe();

    std::ostream_iterator<FieldDescription> osi(std::cout, "\n");
    std::copy(description.begin(), description.end(), osi);
    
    TableBase table(layout, 16);
    
    RecordBase r0 = table.append();
    r0.set(myIntField, 53);    
    r0.set(myArrayField, Eigen::VectorXd::Ones(5));
    r0.set(myFloatField, 3.14);

}
