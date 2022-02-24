using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -0.27492067550218746 <= q <= 2.084201684402027)

                     add_NL_constraint(m, :((-0.0416569430659357 + 0.13675627334278007*(0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + 0.703085236566185*(-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.0416569430659357 + 0.13675627334278007*(0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + 0.703085236566185*(-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + (0.6991595157542867 + 0.5132010992296534*(0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + 0.6682334830395109*(-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.6991595157542867 + 0.5132010992296534*(0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.1752604319892206 + -0.8557289632032381*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.21149394336428795*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + 0.6682334830395109*(-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.4914467550189556 + 0.5078124686195609*(-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.028885055816546323 + 0.15091441453709065*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + -0.8760277243880852*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.7523099505151869*(-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)*(1 + erf((-0.29077418744555983 + -0.11119024401630329*(-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))*(1 + erf((-0.5352161553133312 + 0.05336126885466008*$(x[1]) + 0.7318437425770079*$(x[2]) + -0.7787417630340121*$(x[3]))/sqrt(2)))/2 + 0.4817613487326513*(0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))*(1 + erf((0.0006254708492061667 + 0.44360632638752495*$(x[1]) + -0.9819405794216607*$(x[2]) + 0.2295552892872319*$(x[3]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    