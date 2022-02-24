using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -1.263105417837706 <= q <= 2.4766737340949954)

                     add_NL_constraint(m, :(swish(0.7635571507341847 + -0.35250618093605723*swish(0.5051094873509578 + -0.17934163301535788*swish(-0.331542222824047 + 0.7450661875335634*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + -0.10668129539960924*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4]))) + -0.9610191438924418*swish(0.6781688636512797 + -0.22580923517577522*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + 0.2965219706242044*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4])))) + 0.1709251594821839*swish(0.058205314222445015 + 0.7954955842799771*swish(-0.331542222824047 + 0.7450661875335634*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + -0.10668129539960924*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4]))) + -0.9762293687340509*swish(0.6781688636512797 + -0.22580923517577522*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + 0.2965219706242044*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4]))))) + swish(-0.29663194049169395 + 0.013502946151282558*swish(0.5051094873509578 + -0.17934163301535788*swish(-0.331542222824047 + 0.7450661875335634*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + -0.10668129539960924*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4]))) + -0.9610191438924418*swish(0.6781688636512797 + -0.22580923517577522*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + 0.2965219706242044*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4])))) + -0.2459643040595232*swish(0.058205314222445015 + 0.7954955842799771*swish(-0.331542222824047 + 0.7450661875335634*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + -0.10668129539960924*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4]))) + -0.9762293687340509*swish(0.6781688636512797 + -0.22580923517577522*swish(0.04378671584682481 + 0.6928572029860351*$(x[1]) + 0.8337172801246142*$(x[2]) + 0.1196869024103484*$(x[3]) + -0.7815194586429408*$(x[4])) + 0.2965219706242044*swish(0.8574467662893772 + -0.06659837723966744*$(x[1]) + -0.15959391344635332*$(x[2]) + -0.6414263171723844*$(x[3]) + 0.9253455779716249*$(x[4]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    