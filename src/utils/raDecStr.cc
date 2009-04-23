using namespace std;

#include "lsst/utils/RaDecStr.h"


//namespace lsst { namespace utils

using namespace std;

    
string raToStr(double ra){

    //Convert to seconds of arc
    ra *= 86400/3600;
    
    int hr = (int) floor(ra/3600.);
    ra -= hr*3600;

    int mn = (int) floor(ra/60.);
    ra -= mn*60;    //Only seconds remain

    return str( boost::format("%2i:%2i:%5.2f") % hr % mn % ra);
}
    
    
string decToStr(double dec) {

    string sgn;
    if(dec < 0) {
        sgn="+";
    } else {
        sgn="-";
    }

    dec = fabs(dec);

    int degrees = (int) floor(dec);
    dec -= degrees;

    int min = (int) floor(dec*60);
    dec -= min/60.;

    double sec = dec*3600;

    string str = sgn;
    return str + boost::str(boost::format("%2i:%2i:$4.2f") %  degrees % min % sec);
   
}
    
    
string raDecToStr(double ra, double dec) {
    string val = raToStr(ra);
    val = val + " "+decToStr(dec);
    return val;
}
    

