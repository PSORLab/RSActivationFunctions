using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -37.14009556780016 <= q <= 48.365016418887166)

                     add_NL_constraint(m, :(swish(0.24902435571631587 + -0.02312729681609138*swish(-0.46870412880299783 + 0.1700038462852933*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.44738483682757924*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.5112184633122538*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.17714072857386487*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9529511507705823*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.5418072450548097*swish(0.5019192867350393 + 0.2997113620022893*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.5721484689942011*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.4073624372378828*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + -0.07630883358086304*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.3103910161241421*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.9789829058919253*swish(-0.652369829242545 + -0.998242239199818*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.041041478696029365*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.4348093907280841*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.7773011179238121*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7537599741964742*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.13446361488468206*swish(-0.40141742757451926 + 0.7289092837988314*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + 0.16077592157977172*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.6378484913808045*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9326240829024273*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9638370498324353*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.3069132295129271*swish(0.17680227033398044 + 0.8293506390949057*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.09356004658531791*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.16358082594109513*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9029174038261001*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7318887196209252*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3])))) + swish(-0.017552134571214406 + -0.41004797399293524*swish(-0.46870412880299783 + 0.1700038462852933*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.44738483682757924*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.5112184633122538*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.17714072857386487*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9529511507705823*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.47019448283071075*swish(0.5019192867350393 + 0.2997113620022893*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.5721484689942011*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.4073624372378828*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + -0.07630883358086304*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.3103910161241421*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.35795867979675133*swish(-0.652369829242545 + -0.998242239199818*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.041041478696029365*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.4348093907280841*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.7773011179238121*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7537599741964742*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.44715886126593407*swish(-0.40141742757451926 + 0.7289092837988314*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + 0.16077592157977172*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.6378484913808045*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9326240829024273*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9638370498324353*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.22638221307132733*swish(0.17680227033398044 + 0.8293506390949057*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.09356004658531791*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.16358082594109513*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9029174038261001*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7318887196209252*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3])))) + swish(-0.4029592795849051 + -0.6197606146665708*swish(-0.46870412880299783 + 0.1700038462852933*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.44738483682757924*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.5112184633122538*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.17714072857386487*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9529511507705823*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.8464388391219866*swish(0.5019192867350393 + 0.2997113620022893*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.5721484689942011*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.4073624372378828*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + -0.07630883358086304*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.3103910161241421*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.6262663939568953*swish(-0.652369829242545 + -0.998242239199818*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.041041478696029365*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.4348093907280841*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.7773011179238121*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7537599741964742*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.06090941414894013*swish(-0.40141742757451926 + 0.7289092837988314*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + 0.16077592157977172*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.6378484913808045*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9326240829024273*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9638370498324353*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.6699649674430836*swish(0.17680227033398044 + 0.8293506390949057*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.09356004658531791*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.16358082594109513*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9029174038261001*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7318887196209252*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3])))) + swish(0.6579858782021573 + -0.21911632564812766*swish(-0.46870412880299783 + 0.1700038462852933*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.44738483682757924*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.5112184633122538*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.17714072857386487*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9529511507705823*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.7292607357990395*swish(0.5019192867350393 + 0.2997113620022893*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.5721484689942011*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.4073624372378828*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + -0.07630883358086304*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.3103910161241421*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.23882038501163416*swish(-0.652369829242545 + -0.998242239199818*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.041041478696029365*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.4348093907280841*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.7773011179238121*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7537599741964742*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.6855544316424598*swish(-0.40141742757451926 + 0.7289092837988314*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + 0.16077592157977172*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.6378484913808045*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9326240829024273*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9638370498324353*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.2868272308560216*swish(0.17680227033398044 + 0.8293506390949057*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.09356004658531791*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.16358082594109513*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9029174038261001*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7318887196209252*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3])))) + swish(-0.10147744083820065 + -0.7175110849623545*swish(-0.46870412880299783 + 0.1700038462852933*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.44738483682757924*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.5112184633122538*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.17714072857386487*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9529511507705823*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.9583278917345623*swish(0.5019192867350393 + 0.2997113620022893*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.5721484689942011*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.4073624372378828*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + -0.07630883358086304*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.3103910161241421*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.7313133539296679*swish(-0.652369829242545 + -0.998242239199818*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.041041478696029365*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.4348093907280841*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.7773011179238121*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7537599741964742*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + 0.008349755673621395*swish(-0.40141742757451926 + 0.7289092837988314*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + 0.16077592157977172*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + 0.6378484913808045*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9326240829024273*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.9638370498324353*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3]))) + -0.13663535205769417*swish(0.17680227033398044 + 0.8293506390949057*swish(-0.07115429328131828 + -0.06282459839957566*$(x[1]) + 0.4662931752239805*$(x[2]) + -0.47993632100881634*$(x[3])) + -0.09356004658531791*swish(-0.42367185224307535 + 0.829552620135646*$(x[1]) + -0.7267035823407642*$(x[2]) + 0.6637528273154039*$(x[3])) + -0.16358082594109513*swish(0.6408115001973855 + -0.9252249601617657*$(x[1]) + 0.6902522661180681*$(x[2]) + 0.6531628217744303*$(x[3])) + 0.9029174038261001*swish(0.3778204937005931 + 0.21019890124389518*$(x[1]) + -0.5904566970281198*$(x[2]) + -0.06968695857557394*$(x[3])) + 0.7318887196209252*swish(0.7716365722893217 + 0.12636272698834716*$(x[1]) + 0.15699019471654063*$(x[2]) + -0.5007466869555728*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    