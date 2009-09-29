// -*- lsst-c++ -*-
#if !defined(LSST_AFW_MATH_INTGKPDATA10_H)
#define LSST_AFW_MATH_INTGKPDATA10_H 1

// Gauss-Kronrod-Patterson quadrature coefficients for use in
// quadpack routine qng. These coefficients were calculated with
// 101 decimal digit arithmetic by L. W. Fullerton, Bell Labs, Nov
// 1981. 

#include <vector>

namespace lsst { namespace afw { namespace math {

  namespace details {            
            
  static const int NGKPLEVELS = 5;

  inline int gkp_n(int level) 
  { 
    assert(level >= 0 && level < NGKPLEVELS);
    static const int ngkp[NGKPLEVELS] = {5,5,11,22,44};
    return ngkp[level]; 
  }

  template <class T> inline const std::vector<T>& gkp_x(int level)
  {
    // x10, abscissae common to the 10-, 21-, 43- and 87-point rule 
    static const T ax10[5] = {
      0.973906528517171720077964012084452,
      0.865063366688984510732096688423493,
      0.679409568299024406234327365114874,
      0.433395394129247190799265943165784,
      0.148874338981631210884826001129720
    };
    static const std::vector<T> vx10(ax10,ax10+5);

    // x21, abscissae common to the 21-, 43- and 87-point rule 
    static const T ax21[5] = {
      0.995657163025808080735527280689003,
      0.930157491355708226001207180059508,
      0.780817726586416897063717578345042,
      0.562757134668604683339000099272694,
      0.294392862701460198131126603103866
    };
    static const std::vector<T> vx21(ax21,ax21+5);

    // x43, abscissae common to the 43- and 87-point rule 
    static const T ax43[11] = {
      0.999333360901932081394099323919911,
      0.987433402908088869795961478381209,
      0.954807934814266299257919200290473,
      0.900148695748328293625099494069092,
      0.825198314983114150847066732588520,
      0.732148388989304982612354848755461,
      0.622847970537725238641159120344323,
      0.499479574071056499952214885499755,
      0.364901661346580768043989548502644,
      0.222254919776601296498260928066212,
      0.074650617461383322043914435796506
    };
    static const std::vector<T> vx43(ax43,ax43+11);

    // x87, abscissae of the 87-point rule 
    static const T ax87[22] = {
      0.999902977262729234490529830591582,
      0.997989895986678745427496322365960,
      0.992175497860687222808523352251425,
      0.981358163572712773571916941623894,
      0.965057623858384619128284110607926,
      0.943167613133670596816416634507426,
      0.915806414685507209591826430720050,
      0.883221657771316501372117548744163,
      0.845710748462415666605902011504855,
      0.803557658035230982788739474980964,
      0.757005730685495558328942793432020,
      0.706273209787321819824094274740840,
      0.651589466501177922534422205016736,
      0.593223374057961088875273770349144,
      0.531493605970831932285268948562671,
      0.466763623042022844871966781659270,
      0.399424847859218804732101665817923,
      0.329874877106188288265053371824597,
      0.258503559202161551802280975429025,
      0.185695396568346652015917141167606,
      0.111842213179907468172398359241362,
      0.037352123394619870814998165437704
    };
    static const std::vector<T> vx87(ax87,ax87+22);

    // x175, new abscissae of the 175-point rule
    static const T ax175[44] = {
      0.9999863601049729677997719,
      0.9996988005421428623760900,
      0.9987732473138838303043008,
      0.9969583851613681945576954,
      0.9940679446543307664316419,
      0.9899673391741082769960501,
      0.9845657252243253759726856,
      0.9778061574856414566055911,
      0.9696573134362640567828241,
      0.9601075259508299668299148,
      0.9491605191933640559839528,
      0.9368321320434676982049391,
      0.9231475266471484946315312,
      0.9081385954543932335481458,
      0.8918414554201179766275301,
      0.8742940658668577971868106,
      0.8555341249920432726468181,
      0.8355974653331171075829366,
      0.8145171513255796294567278,
      0.7923233739697216224661771,
      0.7690440708082055924983192,
      0.7447060413364141221086362,
      0.7193362474393392286563226,
      0.6929630148748741413274435,
      0.6656169582351898081906710,
      0.6373315753624730450361940,
      0.6081435407529742253347887,
      0.5780927509986774074337714,
      0.5472221576559162873479056,
      0.5155774001694817879753989,
      0.4832062516695473663839323,
      0.4501579202236101928554315,
      0.4164822936334372150371142,
      0.3822292525735955063074986,
      0.3474481818229070516859048,
      0.3121877718977455143765043,
      0.2764961311348905899785770,
      0.2404211453027010909070665,
      0.2040109589145005126324680,
      0.1673144327616352753945014,
      0.1303814601095591675431696,
      0.09326308333660176874846400,
      0.05601141598806823374241435,
      0.01867941779948308845140053
    };
    static const std::vector<T> vx175(ax175,ax175+44);

    static const std::vector<T>* x[NGKPLEVELS] = 
    {&vx10,&vx21,&vx43,&vx87,&vx175};

    assert(level >= 0 && level < NGKPLEVELS);
    return *x[level];
  }

  template <class T> inline const std::vector<T>& gkp_wa(int level)
  {
    // w21a, weights of the 21-point formula for abscissae x10
    static const T aw21a[5] = {
      0.032558162307964727478818972459390,
      0.075039674810919952767043140916190,
      0.109387158802297641899210590325805,
      0.134709217311473325928054001771707,
      0.147739104901338491374841515972068
    };
    static const std::vector<T> vw21a(aw21a,aw21a+5);

    // w43a, weights of the 43-point formula for abscissae x10,x21
    static const T aw43a[10] = {
      0.016296734289666564924281974617663,
      0.037522876120869501461613795898115,
      0.054694902058255442147212685465005,
      0.067355414609478086075553166302174,
      0.073870199632393953432140695251367,
      0.005768556059769796184184327908655,
      0.027371890593248842081276069289151,
      0.046560826910428830743339154433824,
      0.061744995201442564496240336030883,
      0.071387267268693397768559114425516
    };
    static const std::vector<T> vw43a(aw43a,aw43a+10);

    // w87a, weights of the 87-point formula for abscissae x10..x43
    static const T aw87a[21] = {
      0.008148377384149172900002878448190,
      0.018761438201562822243935059003794,
      0.027347451050052286161582829741283,
      0.033677707311637930046581056957588,
      0.036935099820427907614589586742499,
      0.002884872430211530501334156248695,
      0.013685946022712701888950035273128,
      0.023280413502888311123409291030404,
      0.030872497611713358675466394126442,
      0.035693633639418770719351355457044,
      0.000915283345202241360843392549948,
      0.005399280219300471367738743391053,
      0.010947679601118931134327826856808,
      0.016298731696787335262665703223280,
      0.021081568889203835112433060188190,
      0.025370969769253827243467999831710,
      0.029189697756475752501446154084920,
      0.032373202467202789685788194889595,
      0.034783098950365142750781997949596,
      0.036412220731351787562801163687577,
      0.037253875503047708539592001191226
    };
    static const std::vector<T> vw87a(aw87a,aw87a+21);

    // w175a, weights of the 175-point formula for abscissae x10..x87
    static const T aw175a[43] = {
      0.004074188692082600103872341,
      0.009380719100781411819741177,
      0.01367372552502614309257226,
      0.01683885365581896502443513,
      0.01846754991021395380770806,
      0.001442436294030254510518689,
      0.006842973011356375224430623,
      0.01164020675144415562952859,
      0.01543624880585667934076834,
      0.01784681681970938536027839,
      0.0004576754584154192941064591,
      0.002699640110136963534043642,
      0.005473839800559775445647306,
      0.008149365848393670964180954,
      0.01054078444460191775302900,
      0.01268548488462691364854167,
      0.01459484887823787625642303,
      0.01618660123360139484467359,
      0.01739154947518257137619173,
      0.01820611036567589378188459,
      0.01862693775152385427017140,
      0.0001363171844264931156688042,
      0.0009035606060432074452289352,
      0.002048434635928091782250346,
      0.003379145025868061092978424,
      0.004774978836099393880602724,
      0.006164723826122346701274159,
      0.007505223673194467726814647,
      0.008774483993121594091281642,
      0.009969018893220443741650500,
      0.01109746798050614328494604,
      0.01216957356300040269315573,
      0.01318725270741960360319771,
      0.01414345539438560032188640,
      0.01502629056404634765715107,
      0.01582337568571996469999659,
      0.01652520670998925164398162,
      0.01712754985211303089259342,
      0.01763120633007834051620255,
      0.01803849481144435059221422,
      0.01834930224922804724856518,
      0.01856027463491628805666919,
      0.01866711437596752016025132
    };
    static const std::vector<T> vw175a(aw175a,aw175a+43);

    static const std::vector<T>* wa[NGKPLEVELS] = 
    {0,&vw21a,&vw43a,&vw87a,&vw175a};

    assert(level >= 1 && level < NGKPLEVELS);
    return *wa[level];
  }

  template <class T> inline const std::vector<T>& gkp_wb(int level)
  {
    // w10, weights of the 10-point formula 
    static const T aw10b[6] = {
      0.066671344308688137593568809893332,
      0.149451349150580593145776339657697,
      0.219086362515982043995534934228163,
      0.269266719309996355091226921569469,
      0.295524224714752870173892994651338,
      0.0
    };
    static const std::vector<T> vw10b(aw10b,aw10b+6);

    // w21b, weights of the 21-point formula for abscissae x21
    static const T aw21b[6] = {
      0.011694638867371874278064396062192,
      0.054755896574351996031381300244580,
      0.093125454583697605535065465083366,
      0.123491976262065851077958109831074,
      0.142775938577060080797094273138717,
      0.149445554002916905664936468389821
    };
    static const std::vector<T> vw21b(aw21b,aw21b+6);

    // w43b, weights of the 43-point formula for abscissae x43
    static const T aw43b[12] = {
      0.001844477640212414100389106552965,
      0.010798689585891651740465406741293,
      0.021895363867795428102523123075149,
      0.032597463975345689443882222526137,
      0.042163137935191811847627924327955,
      0.050741939600184577780189020092084,
      0.058379395542619248375475369330206,
      0.064746404951445885544689259517511,
      0.069566197912356484528633315038405,
      0.072824441471833208150939535192842,
      0.074507751014175118273571813842889,
      0.074722147517403005594425168280423
    };
    static const std::vector<T> vw43b(aw43b,aw43b+12);

    // w87b, weights of the 87-point formula for abscissae x87
    static const T aw87b[23] = {
      0.000274145563762072350016527092881,
      0.001807124155057942948341311753254,
      0.004096869282759164864458070683480,
      0.006758290051847378699816577897424,
      0.009549957672201646536053581325377,
      0.012329447652244853694626639963780,
      0.015010447346388952376697286041943,
      0.017548967986243191099665352925900,
      0.019938037786440888202278192730714,
      0.022194935961012286796332102959499,
      0.024339147126000805470360647041454,
      0.026374505414839207241503786552615,
      0.028286910788771200659968002987960,
      0.030052581128092695322521110347341,
      0.031646751371439929404586051078883,
      0.033050413419978503290785944862689,
      0.034255099704226061787082821046821,
      0.035262412660156681033782717998428,
      0.036076989622888701185500318003895,
      0.036698604498456094498018047441094,
      0.037120549269832576114119958413599,
      0.037334228751935040321235449094698,
      0.037361073762679023410321241766599
    };
    static const std::vector<T> vw87b(aw87b,aw87b+23);

    // w175b, weights of the 175-point formula for abscissae x175
    static const T aw175b[45] = {
      0.00003901440664395060055484226,
      0.0002787312407993348139961464,
      0.0006672934146291138823512275,
      0.001163051749910003055308177,
      0.001738533916888715565420637,
      0.002369555040550195269486165,
      0.003036735036032612976379865,
      0.003725394125884235586678819,
      0.004424387541776355951933433,
      0.005125062882078508655875582,
      0.005820601038952832388647054,
      0.006505667785357412495794539,
      0.007176258976795165213169074,
      0.007829642423806550517967042,
      0.008464316525043036555640657,
      0.009079917879101951192512711,
      0.009677029304308397008134699,
      0.01025687419430508685393322,
      0.01082092935368957685380896,
      0.01137052956179654620079332,
      0.01190655150082017425918844,
      0.01242924137407215963971061,
      0.01293819980598456566872951,
      0.01343248644726851711632966,
      0.01391078107864287587682982,
      0.01437154592198526807352501,
      0.01481316256915705351874381,
      0.01523404482122031244940129,
      0.01563274056894636856297717,
      0.01600802989372176552081826,
      0.01635901148521841710605397,
      0.01668515675646806540972517,
      0.01698630892029276765404134,
      0.01726261506199017604299177,
      0.01751439915887340284018269,
      0.01774200489411242260549908,
      0.01794564958245060930824294,
      0.01812532816637054898168673,
      0.01828078919412312732671357,
      0.01841158014800801587767163,
      0.01851713825962505903167768,
      0.01859689376712028821908313,
      0.01865035760791320287947920,
      0.01867717982648033250824110,
      0.01868053688133951170552407
    };
    static const std::vector<T> vw175b(aw175b,aw175b+45);

    static const std::vector<T>* wb[NGKPLEVELS] = 
    {&vw10b,&vw21b,&vw43b,&vw87b,&vw175b};

    assert(level >= 0 && level < NGKPLEVELS);
    return *wb[level];
  }

#ifdef EXTRA_PREC_H

  template<> inline const std::vector<Quad>& gkp_x<Quad>(int level)
  {
    // x10, abscissae common to the 10-, 21-, 43- and 87-point rule 
    static const Quad ax10[5] = {
      "0.973906528517171720077964012084452",
      "0.865063366688984510732096688423493",
      "0.679409568299024406234327365114874",
      "0.433395394129247190799265943165784",
      "0.148874338981631210884826001129720"
    };
    static const std::vector<Quad> vx10(ax10,ax10+5);

    // x21, abscissae common to the 21-, 43- and 87-point rule 
    static const Quad ax21[5] = {
      "0.995657163025808080735527280689003",
      "0.930157491355708226001207180059508",
      "0.780817726586416897063717578345042",
      "0.562757134668604683339000099272694",
      "0.294392862701460198131126603103866"
    };
    static const std::vector<Quad> vx21(ax21,ax21+5);

    // x43, abscissae common to the 43- and 87-point rule 
    static const Quad ax43[11] = {
      "0.999333360901932081394099323919911",
      "0.987433402908088869795961478381209",
      "0.954807934814266299257919200290473",
      "0.900148695748328293625099494069092",
      "0.825198314983114150847066732588520",
      "0.732148388989304982612354848755461",
      "0.622847970537725238641159120344323",
      "0.499479574071056499952214885499755",
      "0.364901661346580768043989548502644",
      "0.222254919776601296498260928066212",
      "0.074650617461383322043914435796506"
    };
    static const std::vector<Quad> vx43(ax43,ax43+11);

    // x87, abscissae of the 87-point rule 
    static const Quad ax87[22] = {
      "0.999902977262729234490529830591582",
      "0.997989895986678745427496322365960",
      "0.992175497860687222808523352251425",
      "0.981358163572712773571916941623894",
      "0.965057623858384619128284110607926",
      "0.943167613133670596816416634507426",
      "0.915806414685507209591826430720050",
      "0.883221657771316501372117548744163",
      "0.845710748462415666605902011504855",
      "0.803557658035230982788739474980964",
      "0.757005730685495558328942793432020",
      "0.706273209787321819824094274740840",
      "0.651589466501177922534422205016736",
      "0.593223374057961088875273770349144",
      "0.531493605970831932285268948562671",
      "0.466763623042022844871966781659270",
      "0.399424847859218804732101665817923",
      "0.329874877106188288265053371824597",
      "0.258503559202161551802280975429025",
      "0.185695396568346652015917141167606",
      "0.111842213179907468172398359241362",
      "0.037352123394619870814998165437704"
    };
    static const std::vector<Quad> vx87(ax87,ax87+22);

    // x175, new abscissae of the 175-point rule
    static const Quad ax175[44] = {
      "0.9999863601049729677997719",
      "0.9996988005421428623760900",
      "0.9987732473138838303043008",
      "0.9969583851613681945576954",
      "0.9940679446543307664316419",
      "0.9899673391741082769960501",
      "0.9845657252243253759726856",
      "0.9778061574856414566055911",
      "0.9696573134362640567828241",
      "0.9601075259508299668299148",
      "0.9491605191933640559839528",
      "0.9368321320434676982049391",
      "0.9231475266471484946315312",
      "0.9081385954543932335481458",
      "0.8918414554201179766275301",
      "0.8742940658668577971868106",
      "0.8555341249920432726468181",
      "0.8355974653331171075829366",
      "0.8145171513255796294567278",
      "0.7923233739697216224661771",
      "0.7690440708082055924983192",
      "0.7447060413364141221086362",
      "0.7193362474393392286563226",
      "0.6929630148748741413274435",
      "0.6656169582351898081906710",
      "0.6373315753624730450361940",
      "0.6081435407529742253347887",
      "0.5780927509986774074337714",
      "0.5472221576559162873479056",
      "0.5155774001694817879753989",
      "0.4832062516695473663839323",
      "0.4501579202236101928554315",
      "0.4164822936334372150371142",
      "0.3822292525735955063074986",
      "0.3474481818229070516859048",
      "0.3121877718977455143765043",
      "0.2764961311348905899785770",
      "0.2404211453027010909070665",
      "0.2040109589145005126324680",
      "0.1673144327616352753945014",
      "0.1303814601095591675431696",
      "0.09326308333660176874846400",
      "0.05601141598806823374241435",
      "0.01867941779948308845140053"
    };
    static const std::vector<Quad> vx175(ax175,ax175+44);

    static const std::vector<Quad>* x[NGKPLEVELS] = 
    {&vx10,&vx21,&vx43,&vx87,&vx175};

    assert(level >= 0 && level < NGKPLEVELS);
    return *x[level];
  }

  template<> inline const std::vector<Quad>& gkp_wa<Quad>(int level)
  {
    // w21a, weights of the 21-point formula for abscissae x10
    static const Quad aw21a[5] = {
      "0.032558162307964727478818972459390",
      "0.075039674810919952767043140916190",
      "0.109387158802297641899210590325805",
      "0.134709217311473325928054001771707",
      "0.147739104901338491374841515972068"
    };
    static const std::vector<Quad> vw21a(aw21a,aw21a+5);

    // w43a, weights of the 43-point formula for abscissae x10,x21
    static const Quad aw43a[10] = {
      "0.016296734289666564924281974617663",
      "0.037522876120869501461613795898115",
      "0.054694902058255442147212685465005",
      "0.067355414609478086075553166302174",
      "0.073870199632393953432140695251367",
      "0.005768556059769796184184327908655",
      "0.027371890593248842081276069289151",
      "0.046560826910428830743339154433824",
      "0.061744995201442564496240336030883",
      "0.071387267268693397768559114425516"
    };
    static const std::vector<Quad> vw43a(aw43a,aw43a+10);

    // w87a, weights of the 87-point formula for abscissae x10..x43
    static const Quad aw87a[21] = {
      "0.008148377384149172900002878448190",
      "0.018761438201562822243935059003794",
      "0.027347451050052286161582829741283",
      "0.033677707311637930046581056957588",
      "0.036935099820427907614589586742499",
      "0.002884872430211530501334156248695",
      "0.013685946022712701888950035273128",
      "0.023280413502888311123409291030404",
      "0.030872497611713358675466394126442",
      "0.035693633639418770719351355457044",
      "0.000915283345202241360843392549948",
      "0.005399280219300471367738743391053",
      "0.010947679601118931134327826856808",
      "0.016298731696787335262665703223280",
      "0.021081568889203835112433060188190",
      "0.025370969769253827243467999831710",
      "0.029189697756475752501446154084920",
      "0.032373202467202789685788194889595",
      "0.034783098950365142750781997949596",
      "0.036412220731351787562801163687577",
      "0.037253875503047708539592001191226"
    };
    static const std::vector<Quad> vw87a(aw87a,aw87a+21);

    // w175a, weights of the 175-point formula for abscissae x10..x87
    static const Quad aw175a[43] = {
      "0.004074188692082600103872341",
      "0.009380719100781411819741177",
      "0.01367372552502614309257226",
      "0.01683885365581896502443513",
      "0.01846754991021395380770806",
      "0.001442436294030254510518689",
      "0.006842973011356375224430623",
      "0.01164020675144415562952859",
      "0.01543624880585667934076834",
      "0.01784681681970938536027839",
      "0.0004576754584154192941064591",
      "0.002699640110136963534043642",
      "0.005473839800559775445647306",
      "0.008149365848393670964180954",
      "0.01054078444460191775302900",
      "0.01268548488462691364854167",
      "0.01459484887823787625642303",
      "0.01618660123360139484467359",
      "0.01739154947518257137619173",
      "0.01820611036567589378188459",
      "0.01862693775152385427017140",
      "0.0001363171844264931156688042",
      "0.0009035606060432074452289352",
      "0.002048434635928091782250346",
      "0.003379145025868061092978424",
      "0.004774978836099393880602724",
      "0.006164723826122346701274159",
      "0.007505223673194467726814647",
      "0.008774483993121594091281642",
      "0.009969018893220443741650500",
      "0.01109746798050614328494604",
      "0.01216957356300040269315573",
      "0.01318725270741960360319771",
      "0.01414345539438560032188640",
      "0.01502629056404634765715107",
      "0.01582337568571996469999659",
      "0.01652520670998925164398162",
      "0.01712754985211303089259342",
      "0.01763120633007834051620255",
      "0.01803849481144435059221422",
      "0.01834930224922804724856518",
      "0.01856027463491628805666919",
      "0.01866711437596752016025132"
    };
    static const std::vector<Quad> vw175a(aw175a,aw175a+43);

    static const std::vector<Quad>* wa[NGKPLEVELS] = 
    {0,&vw21a,&vw43a,&vw87a,&vw175a};

    assert(level >= 1 && level < NGKPLEVELS);
    return *wa[level];
  }

  template<> inline const std::vector<Quad>& gkp_wb<Quad>(int level)
  {
    // w10, weights of the 10-point formula 
    static const Quad aw10b[6] = {
      "0.066671344308688137593568809893332",
      "0.149451349150580593145776339657697",
      "0.219086362515982043995534934228163",
      "0.269266719309996355091226921569469",
      "0.295524224714752870173892994651338",
      "0.0"
    };
    static const std::vector<Quad> vw10b(aw10b,aw10b+6);

    // w21b, weights of the 21-point formula for abscissae x21
    static const Quad aw21b[6] = {
      "0.011694638867371874278064396062192",
      "0.054755896574351996031381300244580",
      "0.093125454583697605535065465083366",
      "0.123491976262065851077958109831074",
      "0.142775938577060080797094273138717",
      "0.149445554002916905664936468389821"
    };
    static const std::vector<Quad> vw21b(aw21b,aw21b+6);

    // w43b, weights of the 43-point formula for abscissae x43
    static const Quad aw43b[12] = {
      "0.001844477640212414100389106552965",
      "0.010798689585891651740465406741293",
      "0.021895363867795428102523123075149",
      "0.032597463975345689443882222526137",
      "0.042163137935191811847627924327955",
      "0.050741939600184577780189020092084",
      "0.058379395542619248375475369330206",
      "0.064746404951445885544689259517511",
      "0.069566197912356484528633315038405",
      "0.072824441471833208150939535192842",
      "0.074507751014175118273571813842889",
      "0.074722147517403005594425168280423"
    };
    static const std::vector<Quad> vw43b(aw43b,aw43b+12);

    // w87b, weights of the 87-point formula for abscissae x87
    static const Quad aw87b[23] = {
      "0.000274145563762072350016527092881",
      "0.001807124155057942948341311753254",
      "0.004096869282759164864458070683480",
      "0.006758290051847378699816577897424",
      "0.009549957672201646536053581325377",
      "0.012329447652244853694626639963780",
      "0.015010447346388952376697286041943",
      "0.017548967986243191099665352925900",
      "0.019938037786440888202278192730714",
      "0.022194935961012286796332102959499",
      "0.024339147126000805470360647041454",
      "0.026374505414839207241503786552615",
      "0.028286910788771200659968002987960",
      "0.030052581128092695322521110347341",
      "0.031646751371439929404586051078883",
      "0.033050413419978503290785944862689",
      "0.034255099704226061787082821046821",
      "0.035262412660156681033782717998428",
      "0.036076989622888701185500318003895",
      "0.036698604498456094498018047441094",
      "0.037120549269832576114119958413599",
      "0.037334228751935040321235449094698",
      "0.037361073762679023410321241766599"
    };
    static const std::vector<Quad> vw87b(aw87b,aw87b+23);

    // w175b, weights of the 175-point formula for abscissae x175
    static const Quad aw175b[45] = {
      "0.00003901440664395060055484226",
      "0.0002787312407993348139961464",
      "0.0006672934146291138823512275",
      "0.001163051749910003055308177",
      "0.001738533916888715565420637",
      "0.002369555040550195269486165",
      "0.003036735036032612976379865",
      "0.003725394125884235586678819",
      "0.004424387541776355951933433",
      "0.005125062882078508655875582",
      "0.005820601038952832388647054",
      "0.006505667785357412495794539",
      "0.007176258976795165213169074",
      "0.007829642423806550517967042",
      "0.008464316525043036555640657",
      "0.009079917879101951192512711",
      "0.009677029304308397008134699",
      "0.01025687419430508685393322",
      "0.01082092935368957685380896",
      "0.01137052956179654620079332",
      "0.01190655150082017425918844",
      "0.01242924137407215963971061",
      "0.01293819980598456566872951",
      "0.01343248644726851711632966",
      "0.01391078107864287587682982",
      "0.01437154592198526807352501",
      "0.01481316256915705351874381",
      "0.01523404482122031244940129",
      "0.01563274056894636856297717",
      "0.01600802989372176552081826",
      "0.01635901148521841710605397",
      "0.01668515675646806540972517",
      "0.01698630892029276765404134",
      "0.01726261506199017604299177",
      "0.01751439915887340284018269",
      "0.01774200489411242260549908",
      "0.01794564958245060930824294",
      "0.01812532816637054898168673",
      "0.01828078919412312732671357",
      "0.01841158014800801587767163",
      "0.01851713825962505903167768",
      "0.01859689376712028821908313",
      "0.01865035760791320287947920",
      "0.01867717982648033250824110",
      "0.01868053688133951170552407"
    };
    static const std::vector<Quad> vw175b(aw175b,aw175b+45);

    static const std::vector<Quad>* wb[NGKPLEVELS] = 
    {&vw10b,&vw21b,&vw43b,&vw87b,&vw175b};

    assert(level >= 0 && level < NGKPLEVELS);
    return *wb[level];
  }
#endif

} // end of namespace details

}}} // of namespaces lsst/afw/math

#endif
