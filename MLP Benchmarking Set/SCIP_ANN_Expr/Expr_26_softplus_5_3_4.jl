using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -44.136489287347466 <= q <= 49.69413857436534)

                     add_NL_constraint(m, :(tsoftplus(-0.4629667091495704 + 0.0635748640057705*tsoftplus(-0.6042046752388197 + 0.6592277231568873*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.2208741763724813*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.22903528038627874*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.7010230418017223*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + -0.376498656542668*tsoftplus(0.0906367995339985 + -0.5226270515291551*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.036001934898274524*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.5808237525352946*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + -0.9838144225655032*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + 0.9837528862049827*tsoftplus(-0.6843081901347321 + 0.1759084220502265*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.8244973565557481*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.43408436278746887*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5138538554581582*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + 0.6929870145414805*tsoftplus(0.17466136320289039 + 0.34912399579570064*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.5430189573257249*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + 0.8343827808008264*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5024555830956028*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5])))) + tsoftplus(-0.02843938861035733 + -0.9215608139047524*tsoftplus(-0.6042046752388197 + 0.6592277231568873*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.2208741763724813*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.22903528038627874*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.7010230418017223*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + 0.43055607662668693*tsoftplus(0.0906367995339985 + -0.5226270515291551*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.036001934898274524*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.5808237525352946*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + -0.9838144225655032*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + 0.2739191304743813*tsoftplus(-0.6843081901347321 + 0.1759084220502265*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.8244973565557481*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.43408436278746887*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5138538554581582*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + -0.15174123897012892*tsoftplus(0.17466136320289039 + 0.34912399579570064*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.5430189573257249*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + 0.8343827808008264*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5024555830956028*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5])))) + tsoftplus(0.15302620454359017 + 0.2649851452072789*tsoftplus(-0.6042046752388197 + 0.6592277231568873*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.2208741763724813*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.22903528038627874*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.7010230418017223*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + 0.3544846227401033*tsoftplus(0.0906367995339985 + -0.5226270515291551*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.036001934898274524*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.5808237525352946*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + -0.9838144225655032*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + -0.15393380179627592*tsoftplus(-0.6843081901347321 + 0.1759084220502265*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.8244973565557481*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.43408436278746887*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5138538554581582*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + -0.5021057410866123*tsoftplus(0.17466136320289039 + 0.34912399579570064*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.5430189573257249*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + 0.8343827808008264*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5024555830956028*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5])))) + tsoftplus(-0.02493154814702514 + 0.9036282434573426*tsoftplus(-0.6042046752388197 + 0.6592277231568873*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.2208741763724813*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.22903528038627874*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.7010230418017223*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + -0.8197312498251352*tsoftplus(0.0906367995339985 + -0.5226270515291551*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + 0.036001934898274524*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.5808237525352946*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + -0.9838144225655032*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + -0.8165988170154601*tsoftplus(-0.6843081901347321 + 0.1759084220502265*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.8244973565557481*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + -0.43408436278746887*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5138538554581582*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5]))) + -0.31962633825426146*tsoftplus(0.17466136320289039 + 0.34912399579570064*tsoftplus(0.6241743795913464 + -0.766722090985696*$(x[1]) + -0.7858005730935416*$(x[2]) + -0.6858228740383199*$(x[3]) + -0.6082385892904831*$(x[4]) + 0.9985801029863852*$(x[5])) + -0.5430189573257249*tsoftplus(-0.3447130292292071 + -0.6519485784237862*$(x[1]) + 0.7604864229259465*$(x[2]) + 0.0992734690453334*$(x[3]) + 0.8197798183291933*$(x[4]) + -0.318898973070731*$(x[5])) + 0.8343827808008264*tsoftplus(0.8804275182042329 + -0.8866983282959149*$(x[1]) + -0.2229099058488231*$(x[2]) + -0.09181845951684497*$(x[3]) + -0.9070898482666334*$(x[4]) + 0.40750583341215263*$(x[5])) + 0.5024555830956028*tsoftplus(0.7993381675094278 + -0.14276675386219884*$(x[1]) + 0.6936315227515886*$(x[2]) + 0.7504068752625699*$(x[3]) + 0.2532989106820014*$(x[4]) + 0.7291563194210262*$(x[5])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    