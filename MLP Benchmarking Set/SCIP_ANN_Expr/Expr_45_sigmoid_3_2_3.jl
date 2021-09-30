using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -6.511064574757294 <= q <= 5.892427832823655)

                     add_NL_constraint(m, :(tsigmoid(0.8420667961720896 + 0.6377773008190584*tsigmoid(0.2706430015022474 + 0.37913826195327127*$(x[1]) + -0.23246438771028233*$(x[2]) + -0.962858624853828*$(x[3])) + -0.5318833078719116*tsigmoid(0.41471473946626647 + -0.5233172992770205*$(x[1]) + 0.18979014479859257*$(x[2]) + -0.9241474042889664*$(x[3])) + 0.4444763614985421*tsigmoid(0.8535403589175976 + 0.11518761269408406*$(x[1]) + 0.1743905022261436*$(x[2]) + 0.7905847203390914*$(x[3]))) + tsigmoid(-0.6920118722887656 + -0.004291479494425321*tsigmoid(0.2706430015022474 + 0.37913826195327127*$(x[1]) + -0.23246438771028233*$(x[2]) + -0.962858624853828*$(x[3])) + -0.6659866917526283*tsigmoid(0.41471473946626647 + -0.5233172992770205*$(x[1]) + 0.18979014479859257*$(x[2]) + -0.9241474042889664*$(x[3])) + -0.9737767362931238*tsigmoid(0.8535403589175976 + 0.11518761269408406*$(x[1]) + 0.1743905022261436*$(x[2]) + 0.7905847203390914*$(x[3]))) + tsigmoid(0.49364351080307367 + -0.7627504091402342*tsigmoid(0.2706430015022474 + 0.37913826195327127*$(x[1]) + -0.23246438771028233*$(x[2]) + -0.962858624853828*$(x[3])) + 0.24762036941954424*tsigmoid(0.41471473946626647 + -0.5233172992770205*$(x[1]) + 0.18979014479859257*$(x[2]) + -0.9241474042889664*$(x[3])) + -0.08455427932500648*tsigmoid(0.8535403589175976 + 0.11518761269408406*$(x[1]) + 0.1743905022261436*$(x[2]) + 0.7905847203390914*$(x[3]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    