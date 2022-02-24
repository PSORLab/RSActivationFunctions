using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -88.40110862821636 <= q <= 80.25263273980681)

                     add_NL_constraint(m, :(sigmoid(-0.3402051397109518 + -0.28456924845213605*sigmoid(-0.42610478134649465 + 0.43223652991075534*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.5575457569666615*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.7868073625875285*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.5078849457963623*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.2883931154790984*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.6888193490874222*sigmoid(-0.17528834888704425 + -0.1425247358809414*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + 0.07932201643104397*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.9443705506849667*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.7094815218028745*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.7352206438426769*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.026837640444652866*sigmoid(-0.005280183583000753 + 0.6172883711874637*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.07641401576689111*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.42701726221355685*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + -0.5680942985776078*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.9418726475599537*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.08739699596173622*sigmoid(0.26121381489212014 + -0.018200625348021138*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.03849358234625422*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.856604860683484*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.2143578715735437*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.706261759527687*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.2360509887003217*sigmoid(-0.09690410604182231 + 0.6473593808554208*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.6752519993458765*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.771479525571356*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.8906247556095672*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.560146720685823*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5])))) + sigmoid(0.2351041904461102 + 0.11979579421065711*sigmoid(-0.42610478134649465 + 0.43223652991075534*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.5575457569666615*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.7868073625875285*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.5078849457963623*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.2883931154790984*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.012124046222294904*sigmoid(-0.17528834888704425 + -0.1425247358809414*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + 0.07932201643104397*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.9443705506849667*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.7094815218028745*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.7352206438426769*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.704332995743346*sigmoid(-0.005280183583000753 + 0.6172883711874637*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.07641401576689111*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.42701726221355685*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + -0.5680942985776078*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.9418726475599537*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.7162932434639226*sigmoid(0.26121381489212014 + -0.018200625348021138*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.03849358234625422*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.856604860683484*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.2143578715735437*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.706261759527687*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.3351546298978829*sigmoid(-0.09690410604182231 + 0.6473593808554208*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.6752519993458765*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.771479525571356*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.8906247556095672*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.560146720685823*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5])))) + sigmoid(-0.4731175976624935 + -0.5144698215629462*sigmoid(-0.42610478134649465 + 0.43223652991075534*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.5575457569666615*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.7868073625875285*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.5078849457963623*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.2883931154790984*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.005430014157027596*sigmoid(-0.17528834888704425 + -0.1425247358809414*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + 0.07932201643104397*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.9443705506849667*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.7094815218028745*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.7352206438426769*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.05201762086802919*sigmoid(-0.005280183583000753 + 0.6172883711874637*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.07641401576689111*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.42701726221355685*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + -0.5680942985776078*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.9418726475599537*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.06728299071980626*sigmoid(0.26121381489212014 + -0.018200625348021138*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.03849358234625422*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.856604860683484*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.2143578715735437*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.706261759527687*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.8893971601390014*sigmoid(-0.09690410604182231 + 0.6473593808554208*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.6752519993458765*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.771479525571356*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.8906247556095672*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.560146720685823*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5])))) + sigmoid(-0.8380258290664089 + 0.49770714260784343*sigmoid(-0.42610478134649465 + 0.43223652991075534*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.5575457569666615*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.7868073625875285*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.5078849457963623*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.2883931154790984*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.732919146629087*sigmoid(-0.17528834888704425 + -0.1425247358809414*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + 0.07932201643104397*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.9443705506849667*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.7094815218028745*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.7352206438426769*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.2983691588163282*sigmoid(-0.005280183583000753 + 0.6172883711874637*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.07641401576689111*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.42701726221355685*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + -0.5680942985776078*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.9418726475599537*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.6200389337758314*sigmoid(0.26121381489212014 + -0.018200625348021138*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.03849358234625422*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.856604860683484*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.2143578715735437*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.706261759527687*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.03199113104777407*sigmoid(-0.09690410604182231 + 0.6473593808554208*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.6752519993458765*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.771479525571356*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.8906247556095672*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.560146720685823*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5])))) + sigmoid(0.7168755118620993 + -0.38856981611164576*sigmoid(-0.42610478134649465 + 0.43223652991075534*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.5575457569666615*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.7868073625875285*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.5078849457963623*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.2883931154790984*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.9818525806719265*sigmoid(-0.17528834888704425 + -0.1425247358809414*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + 0.07932201643104397*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.9443705506849667*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.7094815218028745*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + -0.7352206438426769*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.9400532013818674*sigmoid(-0.005280183583000753 + 0.6172883711874637*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.07641401576689111*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.42701726221355685*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + -0.5680942985776078*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.9418726475599537*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + -0.928831604242299*sigmoid(0.26121381489212014 + -0.018200625348021138*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.03849358234625422*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + -0.856604860683484*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.2143578715735437*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.706261759527687*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5]))) + 0.6913287958041021*sigmoid(-0.09690410604182231 + 0.6473593808554208*sigmoid(-0.9360387382859474 + 0.27356363334073697*$(x[1]) + 0.5623428270051978*$(x[2]) + -0.9519487490584724*$(x[3]) + 0.5639775172803199*$(x[4]) + 0.8360124736609604*$(x[5])) + -0.6752519993458765*sigmoid(0.3895701371443887 + 0.5298392176190725*$(x[1]) + -0.9900818182438851*$(x[2]) + 0.9005096697487907*$(x[3]) + 0.008235253700211498*$(x[4]) + 0.03120241694369552*$(x[5])) + 0.771479525571356*sigmoid(0.7802421455771191 + -0.3899284089358539*$(x[1]) + -0.510799262552962*$(x[2]) + -0.16925880893067768*$(x[3]) + -0.7702701325797845*$(x[4]) + -0.21353502867805174*$(x[5])) + 0.8906247556095672*sigmoid(-0.4643333519515682 + -0.743154946296293*$(x[1]) + 0.08142803382004216*$(x[2]) + 0.8067509325852682*$(x[3]) + -0.6642228946665827*$(x[4]) + -0.03791136441363552*$(x[5])) + 0.560146720685823*sigmoid(0.6488423858989929 + -0.3655334329180211*$(x[1]) + 0.8449912783652063*$(x[2]) + -0.7635680273216123*$(x[3]) + 0.17014039681297044*$(x[4]) + -0.14496676502055283*$(x[5])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    