using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -10.738123900389908 <= q <= 10.990915091916891)

                     add_NL_constraint(m, :(tsigmoid(0.6682820373243992 + -0.7455034136534024*tsigmoid(0.3869002487983626 + 0.20359315376007725*$(x[1]) + -0.5862002654731504*$(x[2]) + -0.4104146539473885*$(x[3])) + 0.6762485340296798*tsigmoid(-0.8119725784580232 + 0.16543252823537946*$(x[1]) + -0.5798934585630247*$(x[2]) + 0.8301587243484687*$(x[3])) + -0.20366258225421552*tsigmoid(-0.7458478227054193 + 0.9695049018594415*$(x[1]) + -0.8216898325776212*$(x[2]) + -0.34749650245920627*$(x[3])) + 0.4376834747265179*tsigmoid(0.760497418452343 + -0.99065623631633*$(x[1]) + 0.7132722774559341*$(x[2]) + -0.9641971541117673*$(x[3]))) + tsigmoid(-0.6082798121856934 + -0.028699055567702914*tsigmoid(0.3869002487983626 + 0.20359315376007725*$(x[1]) + -0.5862002654731504*$(x[2]) + -0.4104146539473885*$(x[3])) + 0.556724097538726*tsigmoid(-0.8119725784580232 + 0.16543252823537946*$(x[1]) + -0.5798934585630247*$(x[2]) + 0.8301587243484687*$(x[3])) + -0.10411350723789292*tsigmoid(-0.7458478227054193 + 0.9695049018594415*$(x[1]) + -0.8216898325776212*$(x[2]) + -0.34749650245920627*$(x[3])) + 0.3271873233246443*tsigmoid(0.760497418452343 + -0.99065623631633*$(x[1]) + 0.7132722774559341*$(x[2]) + -0.9641971541117673*$(x[3]))) + tsigmoid(-0.14186755189642186 + -0.7909536610862462*tsigmoid(0.3869002487983626 + 0.20359315376007725*$(x[1]) + -0.5862002654731504*$(x[2]) + -0.4104146539473885*$(x[3])) + -0.7301822877374571*tsigmoid(-0.8119725784580232 + 0.16543252823537946*$(x[1]) + -0.5798934585630247*$(x[2]) + 0.8301587243484687*$(x[3])) + -0.7127752702063184*tsigmoid(-0.7458478227054193 + 0.9695049018594415*$(x[1]) + -0.8216898325776212*$(x[2]) + -0.34749650245920627*$(x[3])) + -0.002595343107360204*tsigmoid(0.760497418452343 + -0.99065623631633*$(x[1]) + 0.7132722774559341*$(x[2]) + -0.9641971541117673*$(x[3]))) + tsigmoid(0.33520988284674624 + -0.37356255909308933*tsigmoid(0.3869002487983626 + 0.20359315376007725*$(x[1]) + -0.5862002654731504*$(x[2]) + -0.4104146539473885*$(x[3])) + 0.49014345783583524*tsigmoid(-0.8119725784580232 + 0.16543252823537946*$(x[1]) + -0.5798934585630247*$(x[2]) + 0.8301587243484687*$(x[3])) + 0.027697515489944813*tsigmoid(-0.7458478227054193 + 0.9695049018594415*$(x[1]) + -0.8216898325776212*$(x[2]) + -0.34749650245920627*$(x[3])) + 0.14352502175643522*tsigmoid(0.760497418452343 + -0.99065623631633*$(x[1]) + 0.7132722774559341*$(x[2]) + -0.9641971541117673*$(x[3]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    