using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -37.92418043260588 <= q <= 19.457860872935036)

                     add_NL_constraint(m, :(log(1 + exp(-0.5948345021955626 + 0.7909419235855366*log(1 + exp(0.16572281076834727 + 0.7701313689304001*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.3960198610694281*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.006655112331988011*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.5604852483433853*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.5611932928423773*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.9350053029495258*log(1 + exp(0.6292157071886879 + 0.7447123643459479*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.8071073342022652*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.7192221993944883*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1911667575027427*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.30728485701247665*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.6696024550400765*log(1 + exp(-0.5341808090129341 + 0.4032770914369115*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.6061337759397625*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.5534397384266061*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.16208042832895408*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.7376214833764658*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.9733717899660395*log(1 + exp(0.43846326345255404 + -0.02862602361460853*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.5029397949622734*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.20686891235628035*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.9777195341190259*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.2597965376833278*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.07261217388131413*log(1 + exp(0.02496868545483233 + -0.04485855107115411*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.47774406442351136*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.5903487696420049*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1719483424118926*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.8023455576856944*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))))) + log(1 + exp(-0.7044745597894595 + -0.85619817917634*log(1 + exp(0.16572281076834727 + 0.7701313689304001*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.3960198610694281*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.006655112331988011*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.5604852483433853*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.5611932928423773*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.37830231556965854*log(1 + exp(0.6292157071886879 + 0.7447123643459479*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.8071073342022652*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.7192221993944883*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1911667575027427*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.30728485701247665*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.12156626196118481*log(1 + exp(-0.5341808090129341 + 0.4032770914369115*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.6061337759397625*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.5534397384266061*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.16208042832895408*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.7376214833764658*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.3744591738356289*log(1 + exp(0.43846326345255404 + -0.02862602361460853*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.5029397949622734*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.20686891235628035*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.9777195341190259*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.2597965376833278*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.4985780257939494*log(1 + exp(0.02496868545483233 + -0.04485855107115411*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.47774406442351136*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.5903487696420049*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1719483424118926*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.8023455576856944*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))))) + log(1 + exp(-0.9683275825894269 + -0.7637278723892034*log(1 + exp(0.16572281076834727 + 0.7701313689304001*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.3960198610694281*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.006655112331988011*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.5604852483433853*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.5611932928423773*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.9143661573707975*log(1 + exp(0.6292157071886879 + 0.7447123643459479*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.8071073342022652*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.7192221993944883*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1911667575027427*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.30728485701247665*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.7824888723979511*log(1 + exp(-0.5341808090129341 + 0.4032770914369115*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.6061337759397625*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.5534397384266061*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.16208042832895408*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.7376214833764658*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.6866950641423624*log(1 + exp(0.43846326345255404 + -0.02862602361460853*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.5029397949622734*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.20686891235628035*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.9777195341190259*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.2597965376833278*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.09633196432882363*log(1 + exp(0.02496868545483233 + -0.04485855107115411*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.47774406442351136*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.5903487696420049*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1719483424118926*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.8023455576856944*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))))) + log(1 + exp(-0.5143076265809325 + -0.11312975790997548*log(1 + exp(0.16572281076834727 + 0.7701313689304001*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.3960198610694281*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.006655112331988011*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.5604852483433853*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.5611932928423773*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.7215302240627146*log(1 + exp(0.6292157071886879 + 0.7447123643459479*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.8071073342022652*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.7192221993944883*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1911667575027427*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.30728485701247665*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.8436266175918532*log(1 + exp(-0.5341808090129341 + 0.4032770914369115*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.6061337759397625*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.5534397384266061*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.16208042832895408*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.7376214833764658*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.8029468027318871*log(1 + exp(0.43846326345255404 + -0.02862602361460853*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.5029397949622734*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.20686891235628035*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.9777195341190259*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.2597965376833278*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.07363029133667531*log(1 + exp(0.02496868545483233 + -0.04485855107115411*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.47774406442351136*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.5903487696420049*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1719483424118926*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.8023455576856944*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))))) + log(1 + exp(-0.8626839052591899 + -0.6726505975051698*log(1 + exp(0.16572281076834727 + 0.7701313689304001*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.3960198610694281*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.006655112331988011*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.5604852483433853*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.5611932928423773*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + -0.7431517048783043*log(1 + exp(0.6292157071886879 + 0.7447123643459479*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.8071073342022652*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.7192221993944883*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1911667575027427*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.30728485701247665*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.03587478002549904*log(1 + exp(-0.5341808090129341 + 0.4032770914369115*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + -0.6061337759397625*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + 0.5534397384266061*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.16208042832895408*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.7376214833764658*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.7126331133171049*log(1 + exp(0.43846326345255404 + -0.02862602361460853*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.5029397949622734*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.20686891235628035*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.9777195341190259*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.2597965376833278*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))) + 0.7945114361960401*log(1 + exp(0.02496868545483233 + -0.04485855107115411*log(1 + exp(0.4458559142916436 + -0.1548130095620479*$(x[1]) + 0.5123092826047779*$(x[2]))) + 0.47774406442351136*log(1 + exp(0.5904352841714982 + -0.7465945594038805*$(x[1]) + -0.5086687599184327*$(x[2]))) + -0.5903487696420049*log(1 + exp(-0.6551208030223536 + -0.40103431116018706*$(x[1]) + -0.6543625861790279*$(x[2]))) + -0.1719483424118926*log(1 + exp(-0.8605921328376098 + -0.6449201156921633*$(x[1]) + -0.9058095618930082*$(x[2]))) + -0.8023455576856944*log(1 + exp(-0.4569139457142626 + -0.5316223845746624*$(x[1]) + 0.2651376452711478*$(x[2]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    