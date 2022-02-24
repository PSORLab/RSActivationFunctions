using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -12.548880784274985 <= q <= 15.06837792907793)

                     add_NL_constraint(m, :(swish(0.4386422028907555 + 0.03049853723546203*swish(-0.33806544253006265 + -0.5419050015026476*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + -0.12420368558845318*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + 0.008939690927983968*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4]))) + -0.7963936693468932*swish(0.35455126051631725 + -0.3141650319667155*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + 0.26311214619974344*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + -0.4178451532515153*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4]))) + -0.19871150900740808*swish(-0.9712553762218641 + -0.9788234903921222*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + -0.6598676192535482*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + 0.7900118209668339*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4])))) + swish(0.039843038969352484 + 0.2885956651060697*swish(-0.33806544253006265 + -0.5419050015026476*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + -0.12420368558845318*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + 0.008939690927983968*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4]))) + 0.8477392827604406*swish(0.35455126051631725 + -0.3141650319667155*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + 0.26311214619974344*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + -0.4178451532515153*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4]))) + -0.38122333260621444*swish(-0.9712553762218641 + -0.9788234903921222*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + -0.6598676192535482*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + 0.7900118209668339*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4])))) + swish(0.5234531332153054 + -0.26529753551452195*swish(-0.33806544253006265 + -0.5419050015026476*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + -0.12420368558845318*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + 0.008939690927983968*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4]))) + 0.29155449146294243*swish(0.35455126051631725 + -0.3141650319667155*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + 0.26311214619974344*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + -0.4178451532515153*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4]))) + -0.357141816330512*swish(-0.9712553762218641 + -0.9788234903921222*swish(0.3281182697751017 + -0.8099128015858463*$(x[1]) + 0.13317347784439315*$(x[2]) + 0.4405761556905756*$(x[3]) + 0.9454727306320025*$(x[4])) + -0.6598676192535482*swish(-0.11273025364322331 + 0.7707180324977863*$(x[1]) + -0.4030548048256195*$(x[2]) + 0.6515536701116291*$(x[3]) + -0.7265975512301988*$(x[4])) + 0.7900118209668339*swish(-0.1658555453462811 + -0.8928983125267513*$(x[1]) + 0.8779481001349092*$(x[2]) + -0.7642847980322278*$(x[3]) + -0.7283572839362114*$(x[4])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    