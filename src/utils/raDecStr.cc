
// -*- lsst-c++ -*-
#include "lsst/afw/utils/raDecStr.h"


namespace lsst {
    namespace afw {
        namespace utils {
        

using namespace std;

string raToStr(double ra){

    //convert degrees to segasadecimal seconds
    ra *= 86400/360.;

    int hrs = (int) floor(ra/3600.);
    ra -= hrs*3600;
    int mins = (int) floor(ra/60.);
    ra -= mins*60;

    return boost::str(boost::format("%2i:%02i:%05.2f") % hrs % mins % ra);
}
    
            
string decToStr(double dec) {

    string sgn;
    if(dec < 0) {
        sgn="-";
    } else {
        sgn="+";
    }

    dec = fabs(dec);

    int degrees = (int) floor(dec);
    dec -= degrees;

    int min = (int) floor(dec*60);
    dec -= min/60.;

    double sec = dec*3600;

    string str = sgn;
    return str + " " + boost::str(boost::format("%02i:%02i:%04.1f") %  degrees % min % sec);
   
}

string raDecToStr(double ra, double dec){

    string answer = raToStr(ra);
    answer = answer + decToStr(dec);
    return answer;
}

string raDecToStr(const lsst::afw::image::PointD p) {
    return raDecToStr(p.getX(), p.getY());
}


double strToRa(const std::string &str, const std::string &sep){

    const string r = "([\\-\\d]+)"+sep+"(\\d+)"+sep+"([\\.\\d]+)";
    const boost::regex reg(r);
    boost::smatch what;

    if( ! boost::regex_search(str, what, reg) ) {
        string err = boost::str(boost::format("Error parsing ra string  %s")%str);
        throw(LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException, "Error"));
    }

    if( what.size() != 4) {
        string err = boost::str(boost::format("Error parsing ra string  %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);
    }

    double hr, mn, sc;
    try{
       hr = boost::lexical_cast<double>(what[1]);
    } catch(boost::bad_lexical_cast &) {
        string err = boost::str(boost::format("Error parsing hour from %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);
    }
        
    try{
       mn = boost::lexical_cast<double>(what[2]);
    } catch(boost::bad_lexical_cast &) {
        string err = boost::str(boost::format("Error parsing minute from %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);
    }

     try{
       sc = boost::lexical_cast<double>(what[3]);
    } catch(boost::bad_lexical_cast &) {
        string err = boost::str(boost::format("Error parsing second from %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);\
    }


     double ra = (360/24.)*(hr + (mn + sc/60.)/60. );
     return ra;
}

            
double strToDec(const std::string &str, const std::string &sep){

    const string r = "([\\-\\d]+)"+sep+"(\\d+)"+sep+"([\\.\\d]+)";
    const boost::regex reg(r);
    boost::smatch what;

    if( ! boost::regex_search(str, what, reg) ) {
        string err = boost::str(boost::format("Error parsing dec string  %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);
    }

    if( what.size() != 4) {
         string err = boost::str(boost::format("Error parsing dec string  %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);
    }

    double deg, mn, sc;
    try{
       deg = boost::lexical_cast<double>(what[1]);
    } catch(boost::bad_lexical_cast &) {
        string err = boost::str(boost::format("Error parsing degrees from %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);
    }
        
    try{
       mn = boost::lexical_cast<double>(what[2]);
    } catch(boost::bad_lexical_cast &) {
        string err = boost::str(boost::format("Error parsing minute from %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);
    }

     try{
       sc = boost::lexical_cast<double>(what[3]);
    } catch(boost::bad_lexical_cast &) {
        string err = boost::str(boost::format("Error parsing second from %s")%str);
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, str);\
    }

     cout << what[1] << " " << what[2] << " " << what[3] << endl;
     cout << deg << " " << mn << " " << sc << endl;
     double dec = fabs(deg) + (mn + sc/60.)/60. ;

     //Check if the value is less than zero
     const boost::regex reg2("-");
     if( boost::regex_search(boost::lexical_cast<string>(what[1]),  reg2) ) {
         dec *= -1;
     }
     
     return dec;
}
            
}}}
