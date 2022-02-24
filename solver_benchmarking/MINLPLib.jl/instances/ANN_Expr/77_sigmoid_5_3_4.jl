using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -36.7700820821577 <= q <= 37.6713142442229)

                     add_NL_constraint(m, :(1/(1 + exp(-(0.9266287961121966 + 0.9630341445616355*1/(1 + exp(-(0.9080159995418975 + 0.10769169268185319*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.2597058016512084*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.8597063403720888*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.17950756837409587*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + 0.8197021439185934*1/(1 + exp(-(-0.9264374759333807 + -0.7028105280156569*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.6656745963467641*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.9099166873549671*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.9082646170966706*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + -0.8564198456770362*1/(1 + exp(-(-0.7118497063916944 + 0.36586202274128254*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + -0.9129427463398758*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.5191517354247042*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.42477634545139953*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + 0.7868467121279954*1/(1 + exp(-(0.8429853377330745 + -0.6377098680254001*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.04468698484435718*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.3607403638969684*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.4004266402265344*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5])))))))))) + 1/(1 + exp(-(-0.9209290943615045 + -0.04560727661330999*1/(1 + exp(-(0.9080159995418975 + 0.10769169268185319*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.2597058016512084*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.8597063403720888*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.17950756837409587*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + -0.04187997574232316*1/(1 + exp(-(-0.9264374759333807 + -0.7028105280156569*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.6656745963467641*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.9099166873549671*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.9082646170966706*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + 0.009897273687461627*1/(1 + exp(-(-0.7118497063916944 + 0.36586202274128254*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + -0.9129427463398758*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.5191517354247042*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.42477634545139953*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + -0.4138607422831795*1/(1 + exp(-(0.8429853377330745 + -0.6377098680254001*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.04468698484435718*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.3607403638969684*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.4004266402265344*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5])))))))))) + 1/(1 + exp(-(0.41055131347137674 + 0.5074173875637427*1/(1 + exp(-(0.9080159995418975 + 0.10769169268185319*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.2597058016512084*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.8597063403720888*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.17950756837409587*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + 0.7865169798440914*1/(1 + exp(-(-0.9264374759333807 + -0.7028105280156569*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.6656745963467641*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.9099166873549671*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.9082646170966706*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + -0.6322955384333278*1/(1 + exp(-(-0.7118497063916944 + 0.36586202274128254*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + -0.9129427463398758*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.5191517354247042*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.42477634545139953*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + 0.03598168048021799*1/(1 + exp(-(0.8429853377330745 + -0.6377098680254001*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.04468698484435718*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.3607403638969684*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.4004266402265344*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5])))))))))) + 1/(1 + exp(-(-0.7045693885767101 + 0.765322888364711*1/(1 + exp(-(0.9080159995418975 + 0.10769169268185319*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.2597058016512084*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.8597063403720888*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.17950756837409587*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + 0.9087169409161477*1/(1 + exp(-(-0.9264374759333807 + -0.7028105280156569*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.6656745963467641*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.9099166873549671*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.9082646170966706*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + -0.29202574365958966*1/(1 + exp(-(-0.7118497063916944 + 0.36586202274128254*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + -0.9129427463398758*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + -0.5191517354247042*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.42477634545139953*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5]))))))) + -0.14636407330293322*1/(1 + exp(-(0.8429853377330745 + -0.6377098680254001*1/(1 + exp(-(-0.02399021036854254 + -0.4943608904262171*$(x[1]) + -0.3028702711871105*$(x[2]) + 0.7730701583831108*$(x[3]) + -0.5656579689281509*$(x[4]) + 0.08835909373115314*$(x[5])))) + 0.04468698484435718*1/(1 + exp(-(-0.6908874409241781 + -0.9493883109743697*$(x[1]) + 0.1034335229123764*$(x[2]) + 0.3901179347519834*$(x[3]) + -0.16073609758033847*$(x[4]) + 0.052390589356133166*$(x[5])))) + 0.3607403638969684*1/(1 + exp(-(0.7510327333371474 + 0.48933101371416265*$(x[1]) + 0.5299590321169751*$(x[2]) + 0.05546767568522437*$(x[3]) + -0.4663274814822995*$(x[4]) + -0.6567285643500202*$(x[5])))) + 0.4004266402265344*1/(1 + exp(-(-0.27157848416188113 + 0.560003468778171*$(x[1]) + 0.2933473130688298*$(x[2]) + 0.8808820816540863*$(x[3]) + 0.6726368959629845*$(x[4]) + -0.19953521813356856*$(x[5])))))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    