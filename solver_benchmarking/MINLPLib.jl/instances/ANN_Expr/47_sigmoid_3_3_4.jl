using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -24.652335523355834 <= q <= 21.32472335033785)

                     add_NL_constraint(m, :(1/(1 + exp(-(-0.6982139593401175 + -0.19504968807074974*1/(1 + exp(-(0.6793369023557885 + 0.6523875537646115*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.5142038874540145*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.0456439541419269*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6096854328317232*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + -0.41803298572193937*1/(1 + exp(-(-0.7048442383777926 + 0.47781322839334983*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.713369260368808*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.44964139924772173*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.9876898123617992*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + 0.9542284451941447*1/(1 + exp(-(0.9969387928417852 + -0.011395717836928387*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + 0.8103499615759371*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.12667586987008894*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6252342222679741*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + 0.7916680056214345*1/(1 + exp(-(-0.5156629799588863 + -0.22745979325095078*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.8222624364306839*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.4937419854366709*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.1046356011041838*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3])))))))))) + 1/(1 + exp(-(-0.4539405071565561 + 0.5320780591355936*1/(1 + exp(-(0.6793369023557885 + 0.6523875537646115*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.5142038874540145*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.0456439541419269*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6096854328317232*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + -0.8489788623944037*1/(1 + exp(-(-0.7048442383777926 + 0.47781322839334983*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.713369260368808*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.44964139924772173*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.9876898123617992*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + -0.25620536200506905*1/(1 + exp(-(0.9969387928417852 + -0.011395717836928387*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + 0.8103499615759371*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.12667586987008894*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6252342222679741*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + 0.9264345634068367*1/(1 + exp(-(-0.5156629799588863 + -0.22745979325095078*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.8222624364306839*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.4937419854366709*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.1046356011041838*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3])))))))))) + 1/(1 + exp(-(-0.4017849565754932 + 0.7092320156791541*1/(1 + exp(-(0.6793369023557885 + 0.6523875537646115*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.5142038874540145*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.0456439541419269*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6096854328317232*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + -0.20657569919094643*1/(1 + exp(-(-0.7048442383777926 + 0.47781322839334983*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.713369260368808*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.44964139924772173*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.9876898123617992*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + 0.6260976974386958*1/(1 + exp(-(0.9969387928417852 + -0.011395717836928387*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + 0.8103499615759371*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.12667586987008894*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6252342222679741*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + 0.026773003462286393*1/(1 + exp(-(-0.5156629799588863 + -0.22745979325095078*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.8222624364306839*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.4937419854366709*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.1046356011041838*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3])))))))))) + 1/(1 + exp(-(0.8541923459688499 + -0.7979821016654509*1/(1 + exp(-(0.6793369023557885 + 0.6523875537646115*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.5142038874540145*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.0456439541419269*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6096854328317232*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + 0.2172669330681649*1/(1 + exp(-(-0.7048442383777926 + 0.47781322839334983*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.713369260368808*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.44964139924772173*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.9876898123617992*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + -0.6626207135044284*1/(1 + exp(-(0.9969387928417852 + -0.011395717836928387*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + 0.8103499615759371*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.12667586987008894*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + 0.6252342222679741*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3]))))))) + -0.5735396155482761*1/(1 + exp(-(-0.5156629799588863 + -0.22745979325095078*1/(1 + exp(-(0.1880507757579264 + -0.8703531991881865*$(x[1]) + 0.8423062039785143*$(x[2]) + 0.27626604928751597*$(x[3])))) + -0.8222624364306839*1/(1 + exp(-(0.8004597365811503 + 0.4093664270553261*$(x[1]) + -0.7983161406747903*$(x[2]) + -0.755209029087681*$(x[3])))) + -0.4937419854366709*1/(1 + exp(-(-0.8428376848805481 + -0.4971126206674761*$(x[1]) + -0.0245959153233013*$(x[2]) + 0.4386350201089879*$(x[3])))) + -0.1046356011041838*1/(1 + exp(-(-0.26516744777719925 + 0.12012704290515952*$(x[1]) + -0.4833932117044406*$(x[2]) + -0.25804110713494666*$(x[3])))))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    