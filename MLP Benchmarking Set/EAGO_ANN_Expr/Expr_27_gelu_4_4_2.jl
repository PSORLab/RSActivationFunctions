using JuMP, EAGO

                     m = Model()

                     register(m, :tgelu, 1, tgelu, autodiff = true)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -1.9612509804485732 <= q <= 1.8703437645935266)

                     add_NL_constraint(m, :(tgelu(-0.42566117222835853 + -0.787256486478892*tgelu(-0.5332172640171025 + -0.834926145617227*tgelu(-0.19537604688443544 + -0.767413224028032*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.09063378774625752*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4]))) + 0.8241327481798209*tgelu(0.19899161180175273 + 0.878536291667499*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.04550773895855942*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4])))) + 0.48768666396635396*tgelu(0.24328220393904987 + -0.12545946710242095*tgelu(-0.19537604688443544 + -0.767413224028032*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.09063378774625752*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4]))) + -0.3177194612472731*tgelu(0.19899161180175273 + 0.878536291667499*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.04550773895855942*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4]))))) + tgelu(0.5475075694688742 + 0.16999123312713982*tgelu(-0.5332172640171025 + -0.834926145617227*tgelu(-0.19537604688443544 + -0.767413224028032*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.09063378774625752*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4]))) + 0.8241327481798209*tgelu(0.19899161180175273 + 0.878536291667499*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.04550773895855942*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4])))) + 0.8355575179693098*tgelu(0.24328220393904987 + -0.12545946710242095*tgelu(-0.19537604688443544 + -0.767413224028032*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.09063378774625752*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4]))) + -0.3177194612472731*tgelu(0.19899161180175273 + 0.878536291667499*tgelu(-0.6653759294170118 + -0.05975933394003663*$(x[1]) + -0.415160451093461*$(x[2]) + -0.8425890427832803*$(x[3]) + -0.029284193993054153*$(x[4])) + -0.04550773895855942*tgelu(-0.6312532584598132 + -0.23769942288478996*$(x[1]) + -0.5175720220557212*$(x[2]) + 0.7908279540662599*$(x[3]) + -0.7413603946833156*$(x[4]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    