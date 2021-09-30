using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -9.600004859571829 <= q <= 10.685187204362736)

                     add_NL_constraint(m, :(gelu(-0.9080875874443914 + 0.3302823614493646*gelu(0.708322374279081 + 0.8840807921005829*$(x[1]) + 0.5193959794949516*$(x[2]) + 0.13222217638482547*$(x[3]) + 0.13468375196122873*$(x[4])) + -0.35293588674490906*gelu(-0.4013871620777878 + 0.864355145179998*$(x[1]) + -0.9486700933381309*$(x[2]) + -0.5514371983708801*$(x[3]) + 0.8687248660024327*$(x[4])) + 0.210554022161749*gelu(0.26322365314675 + -0.4105282184596306*$(x[1]) + -0.021130343470725066*$(x[2]) + -0.9380863024119726*$(x[3]) + -0.7800300788285459*$(x[4]))) + gelu(0.510532022444401 + -0.3597488732740972*gelu(0.708322374279081 + 0.8840807921005829*$(x[1]) + 0.5193959794949516*$(x[2]) + 0.13222217638482547*$(x[3]) + 0.13468375196122873*$(x[4])) + -0.5136814000223815*gelu(-0.4013871620777878 + 0.864355145179998*$(x[1]) + -0.9486700933381309*$(x[2]) + -0.5514371983708801*$(x[3]) + 0.8687248660024327*$(x[4])) + 0.6334909039088452*gelu(0.26322365314675 + -0.4105282184596306*$(x[1]) + -0.021130343470725066*$(x[2]) + -0.9380863024119726*$(x[3]) + -0.7800300788285459*$(x[4]))) + gelu(-0.571693351378725 + 0.842031681288602*gelu(0.708322374279081 + 0.8840807921005829*$(x[1]) + 0.5193959794949516*$(x[2]) + 0.13222217638482547*$(x[3]) + 0.13468375196122873*$(x[4])) + -0.5394083296276193*gelu(-0.4013871620777878 + 0.864355145179998*$(x[1]) + -0.9486700933381309*$(x[2]) + -0.5514371983708801*$(x[3]) + 0.8687248660024327*$(x[4])) + 0.5689031971828329*gelu(0.26322365314675 + -0.4105282184596306*$(x[1]) + -0.021130343470725066*$(x[2]) + -0.9380863024119726*$(x[3]) + -0.7800300788285459*$(x[4]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    