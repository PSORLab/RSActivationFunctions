using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -47.02611826463841 <= q <= 46.175694332378946)

                     add_NL_constraint(m, :(log(1 + exp(0.14172886209520552 + -0.7326244672203441*log(1 + exp(-0.9862473619768446 + 0.12598046880516867*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.9857820194451663*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3507087497777257*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.16930171271231975*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + 0.6188053648075824*log(1 + exp(-0.8901994655270467 + 0.3517572387180481*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5057680398491988*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15346000401268212*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.000907155550359029*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + 0.748255673773524*log(1 + exp(0.9418689270026719 + 0.23440750676702393*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.9578891219409944*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4332120963929875*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15216088947964046*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + -0.19681615050825796*log(1 + exp(-0.3160427191298667 + -0.1453900085539872*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3370407689400481*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5646332930109117*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4312398718758548*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))))) + log(1 + exp(0.7126847484534728 + 0.5521736820641339*log(1 + exp(-0.9862473619768446 + 0.12598046880516867*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.9857820194451663*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3507087497777257*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.16930171271231975*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + -0.5863621462894022*log(1 + exp(-0.8901994655270467 + 0.3517572387180481*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5057680398491988*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15346000401268212*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.000907155550359029*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + -0.2365722039484086*log(1 + exp(0.9418689270026719 + 0.23440750676702393*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.9578891219409944*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4332120963929875*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15216088947964046*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + -0.47134077688752996*log(1 + exp(-0.3160427191298667 + -0.1453900085539872*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3370407689400481*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5646332930109117*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4312398718758548*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))))) + log(1 + exp(0.3678622626620536 + -0.645924933237755*log(1 + exp(-0.9862473619768446 + 0.12598046880516867*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.9857820194451663*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3507087497777257*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.16930171271231975*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + 0.23980134898526861*log(1 + exp(-0.8901994655270467 + 0.3517572387180481*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5057680398491988*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15346000401268212*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.000907155550359029*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + 0.5641271464877478*log(1 + exp(0.9418689270026719 + 0.23440750676702393*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.9578891219409944*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4332120963929875*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15216088947964046*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + -0.22456183587218126*log(1 + exp(-0.3160427191298667 + -0.1453900085539872*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3370407689400481*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5646332930109117*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4312398718758548*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))))) + log(1 + exp(-0.9861590919257504 + -0.9031060096581198*log(1 + exp(-0.9862473619768446 + 0.12598046880516867*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.9857820194451663*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3507087497777257*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.16930171271231975*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + -0.6492392304855641*log(1 + exp(-0.8901994655270467 + 0.3517572387180481*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5057680398491988*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15346000401268212*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.000907155550359029*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + 0.1540584284169868*log(1 + exp(0.9418689270026719 + 0.23440750676702393*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.9578891219409944*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4332120963929875*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.15216088947964046*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))) + -0.8682585596517867*log(1 + exp(-0.3160427191298667 + -0.1453900085539872*log(1 + exp(-0.32432918385015697 + -0.6977940842525037*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.476420397288412*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.5549201217740469*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.4315951265237059*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.3370407689400481*log(1 + exp(0.2633379906083797 + 0.3509284626355762*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.5977055959243147*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.642253969451728*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.10048605306401859*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + 0.5646332930109117*log(1 + exp(-0.34549165187553976 + 0.5367448324029058*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.7077894233575681*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + -0.4911236080946888*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + -0.5306931393323575*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))) + -0.4312398718758548*log(1 + exp(0.538919292477293 + 0.8088100362525972*log(1 + exp(-0.9860262540836815 + 0.8911186935536937*$(x[1]) + 0.64267061852934*$(x[2]))) + -0.4318478690072691*log(1 + exp(-0.12607269491856643 + -0.6901389567897818*$(x[1]) + -0.5431979251870089*$(x[2]))) + 0.8077402924106214*log(1 + exp(0.4882924486844056 + 0.5602141072539029*$(x[1]) + 0.7222626702909976*$(x[2]))) + 0.12603114956525063*log(1 + exp(-0.9654866490330036 + -0.27186143772128135*$(x[1]) + -0.864410033347212*$(x[2]))))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    