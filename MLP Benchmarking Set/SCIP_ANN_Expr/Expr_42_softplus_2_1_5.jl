using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -5.159341214783611 <= q <= 3.043092673500856)

                     add_NL_constraint(m, :(tsoftplus(0.865523575364918 + -0.5364939172020486*$(x[1]) + -0.058102576192631705*$(x[2])) + tsoftplus(-0.5760820171418506 + 0.3828384633347359*$(x[1]) + -0.34862012124433184*$(x[2])) + tsoftplus(-0.8481356091724721 + -0.5670909619226432*$(x[1]) + -0.6535866047901866*$(x[2])) + tsoftplus(-0.7891700270237578 + -0.7231220770356499*$(x[1]) + 0.2855200029767526*$(x[2])) + tsoftplus(0.2897398073317854 + -0.49039472307924514*$(x[1]) + -0.0554474963640077*$(x[2])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    