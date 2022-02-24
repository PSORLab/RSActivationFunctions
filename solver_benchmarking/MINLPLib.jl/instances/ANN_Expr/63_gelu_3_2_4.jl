using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -19.57782063716216 <= q <= 17.16108472444359)

                     add_NL_constraint(m, :((-0.6503895050792288 + 0.5000448231799557*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + 0.8223864208279696*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + -0.39198040344556784*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + 0.4922116220556738*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.6503895050792288 + 0.5000448231799557*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + 0.8223864208279696*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + -0.39198040344556784*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + 0.4922116220556738*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + (0.1539410450835783 + -0.07568393712605781*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + -0.902375002810281*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + 0.7665443117707627*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + -0.49047645888895497*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)*(1 + erf((0.1539410450835783 + -0.07568393712605781*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + -0.902375002810281*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + 0.7665443117707627*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + -0.49047645888895497*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + (0.6062617117178619 + -0.6503507748090844*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + -0.9378359170676176*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + -0.23896524724444168*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + -0.3347498056547815*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)*(1 + erf((0.6062617117178619 + -0.6503507748090844*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + -0.9378359170676176*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + -0.23896524724444168*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + -0.3347498056547815*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + (-0.5413645906370745 + 0.4055343512605676*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + -0.13388786735965885*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + -0.9731450409901217*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + 0.17849747613123057*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.5413645906370745 + 0.4055343512605676*(0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))*(1 + erf((0.48086890691524875 + 0.8604652607112211*$(x[1]) + -0.9524910094017911*$(x[2]) + -0.8411501540974782*$(x[3]))/sqrt(2)))/2 + -0.13388786735965885*(0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))*(1 + erf((0.4084009324685236 + 0.9106618206937234*$(x[1]) + -0.8345217492048942*$(x[2]) + 0.6009855404051798*$(x[3]))/sqrt(2)))/2 + -0.9731450409901217*(0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))*(1 + erf((0.29882092390692216 + -0.9088484335369627*$(x[1]) + -0.948042223821898*$(x[2]) + 0.5765779510962643*$(x[3]))/sqrt(2)))/2 + 0.17849747613123057*(0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))*(1 + erf((0.9223388106039003 + 0.14028925335959963*$(x[1]) + -0.026802684789215103*$(x[2]) + 0.9753923898070198*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    