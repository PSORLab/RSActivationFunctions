using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -65.76627584345755 <= q <= 59.26265199263469)

                     add_NL_constraint(m, :(sigmoid(-0.4079491356153899 + -0.18134844314194787*sigmoid(0.2817697329190594 + 0.989386077401103*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.6068435500303653*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.8530798012021474*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.8376755183348754*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.6594644406874322*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.2976334456396539*sigmoid(-0.4984715860832294 + -0.6460783499544172*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.0952691245496573*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.5434919689873383*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.2400778955734424*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.904210133814042*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.11940205422195982*sigmoid(-0.9402956128299329 + 0.19522910845028774*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.5236489579886725*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.09802794343243137*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.5359271533353085*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.5362141752531211*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.6350558501595733*sigmoid(-0.842982128122256 + 0.6833459994096582*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.15560340828874075*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.5368387122163134*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.6086259049043772*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.5403730914454505*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.4683577877900529*sigmoid(0.6067063734003204 + 0.8661978259504042*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.06088102390312189*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.9836506901320718*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.7952476421961925*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.6703006410599177*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3])))) + sigmoid(0.9379679817528435 + -0.4660824895766815*sigmoid(0.2817697329190594 + 0.989386077401103*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.6068435500303653*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.8530798012021474*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.8376755183348754*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.6594644406874322*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.32761179552569075*sigmoid(-0.4984715860832294 + -0.6460783499544172*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.0952691245496573*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.5434919689873383*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.2400778955734424*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.904210133814042*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.6880683972685175*sigmoid(-0.9402956128299329 + 0.19522910845028774*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.5236489579886725*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.09802794343243137*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.5359271533353085*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.5362141752531211*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.5888723964207281*sigmoid(-0.842982128122256 + 0.6833459994096582*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.15560340828874075*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.5368387122163134*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.6086259049043772*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.5403730914454505*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.05595383404471299*sigmoid(0.6067063734003204 + 0.8661978259504042*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.06088102390312189*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.9836506901320718*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.7952476421961925*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.6703006410599177*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3])))) + sigmoid(-0.3048668135843893 + -0.3680078347752045*sigmoid(0.2817697329190594 + 0.989386077401103*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.6068435500303653*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.8530798012021474*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.8376755183348754*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.6594644406874322*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.09989706667818732*sigmoid(-0.4984715860832294 + -0.6460783499544172*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.0952691245496573*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.5434919689873383*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.2400778955734424*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.904210133814042*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.6733365176188579*sigmoid(-0.9402956128299329 + 0.19522910845028774*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.5236489579886725*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.09802794343243137*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.5359271533353085*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.5362141752531211*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.7967060490111804*sigmoid(-0.842982128122256 + 0.6833459994096582*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.15560340828874075*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.5368387122163134*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.6086259049043772*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.5403730914454505*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.8483184805894783*sigmoid(0.6067063734003204 + 0.8661978259504042*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.06088102390312189*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.9836506901320718*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.7952476421961925*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.6703006410599177*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3])))) + sigmoid(-0.8260325513973243 + 0.9282935832138137*sigmoid(0.2817697329190594 + 0.989386077401103*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.6068435500303653*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.8530798012021474*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.8376755183348754*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.6594644406874322*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.5968467249691596*sigmoid(-0.4984715860832294 + -0.6460783499544172*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.0952691245496573*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.5434919689873383*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.2400778955734424*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.904210133814042*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.003449906971975558*sigmoid(-0.9402956128299329 + 0.19522910845028774*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.5236489579886725*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.09802794343243137*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.5359271533353085*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.5362141752531211*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.5427745947324003*sigmoid(-0.842982128122256 + 0.6833459994096582*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.15560340828874075*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.5368387122163134*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.6086259049043772*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.5403730914454505*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.4794292841893828*sigmoid(0.6067063734003204 + 0.8661978259504042*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.06088102390312189*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.9836506901320718*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.7952476421961925*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.6703006410599177*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3])))) + sigmoid(-0.3442450542855955 + 0.3679171232252214*sigmoid(0.2817697329190594 + 0.989386077401103*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.6068435500303653*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.8530798012021474*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.8376755183348754*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.6594644406874322*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.4969759600474797*sigmoid(-0.4984715860832294 + -0.6460783499544172*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.0952691245496573*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.5434919689873383*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.2400778955734424*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.904210133814042*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.17345395101703964*sigmoid(-0.9402956128299329 + 0.19522910845028774*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.5236489579886725*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + -0.09802794343243137*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.5359271533353085*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + 0.5362141752531211*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + -0.24637355767613478*sigmoid(-0.842982128122256 + 0.6833459994096582*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + 0.15560340828874075*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.5368387122163134*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + -0.6086259049043772*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.5403730914454505*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3]))) + 0.4183638306902213*sigmoid(0.6067063734003204 + 0.8661978259504042*sigmoid(-0.7838782818766057 + 0.6083304932652234*$(x[1]) + 0.4982278286707449*$(x[2]) + -0.2123905230276213*$(x[3])) + -0.06088102390312189*sigmoid(-0.6220948510874038 + -0.22981743777707297*$(x[1]) + 0.33188126395442463*$(x[2]) + -0.8264122617752747*$(x[3])) + 0.9836506901320718*sigmoid(0.5453301275165048 + -0.2798145416678528*$(x[1]) + -0.9787415760841154*$(x[2]) + 0.5329097417839521*$(x[3])) + 0.7952476421961925*sigmoid(-0.31934053056836653 + 0.3102265855539219*$(x[1]) + -0.9236015463563882*$(x[2]) + -0.5629164596540628*$(x[3])) + -0.6703006410599177*sigmoid(0.17795541107351465 + 0.6245951080916892*$(x[1]) + -0.14381738742917527*$(x[2]) + 0.44898943772247746*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    