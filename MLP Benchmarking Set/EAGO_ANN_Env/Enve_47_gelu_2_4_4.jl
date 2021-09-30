using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -17.947976160803783 <= q <= 17.121941097508078)

                     add_NL_constraint(m, :(gelu(0.4841602385437169 + 0.29478943491117526*gelu(0.1338947468049696 + -0.4467106636839353*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.38998587550253205*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.1511062775586458*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.3677011750111534*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + -0.9854997427106786*gelu(-0.38016469108937656 + 0.31285704848046336*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.7864910644124823*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.4459799190083178*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.06459214966720817*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + -0.8090728291290188*gelu(0.659320592169808 + -0.5970858485457633*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.2429665451326355*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8651788676931935*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8607640464268691*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + -0.5805626749668584*gelu(-0.7804915801328374 + 0.8608832826393833*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8784429916098833*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.5629057203696992*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.46194713364437323*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))))) + gelu(-0.17229029227549253 + -0.45894422966016224*gelu(0.1338947468049696 + -0.4467106636839353*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.38998587550253205*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.1511062775586458*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.3677011750111534*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + 0.5667158521441267*gelu(-0.38016469108937656 + 0.31285704848046336*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.7864910644124823*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.4459799190083178*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.06459214966720817*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + 0.5550136816913165*gelu(0.659320592169808 + -0.5970858485457633*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.2429665451326355*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8651788676931935*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8607640464268691*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + 0.11061600925746218*gelu(-0.7804915801328374 + 0.8608832826393833*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8784429916098833*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.5629057203696992*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.46194713364437323*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))))) + gelu(0.673317373808417 + 0.2928641141990629*gelu(0.1338947468049696 + -0.4467106636839353*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.38998587550253205*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.1511062775586458*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.3677011750111534*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + 0.6646498863975925*gelu(-0.38016469108937656 + 0.31285704848046336*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.7864910644124823*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.4459799190083178*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.06459214966720817*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + -0.025020242538482673*gelu(0.659320592169808 + -0.5970858485457633*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.2429665451326355*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8651788676931935*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8607640464268691*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + 0.4025709653888554*gelu(-0.7804915801328374 + 0.8608832826393833*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8784429916098833*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.5629057203696992*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.46194713364437323*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))))) + gelu(0.06435126037541394 + 0.8179040879139539*gelu(0.1338947468049696 + -0.4467106636839353*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.38998587550253205*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.1511062775586458*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.3677011750111534*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + 0.2868246388028415*gelu(-0.38016469108937656 + 0.31285704848046336*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.7864910644124823*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.4459799190083178*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.06459214966720817*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + -0.087088206661881*gelu(0.659320592169808 + -0.5970858485457633*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.2429665451326355*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8651788676931935*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8607640464268691*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2])))) + -0.915753588276035*gelu(-0.7804915801328374 + 0.8608832826393833*gelu(0.3789950760560967 + -0.7804156806715459*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.20534015813868312*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.042687005237724396*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.09824171495536449*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + -0.8784429916098833*gelu(-0.35968445527957416 + 0.6574967871598454*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.687421153048589*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + 0.3210131677848622*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + -0.5257521314695688*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.5629057203696992*gelu(0.9349204556200235 + -0.5242015131938644*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + -0.12151261442153016*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.2207883866350664*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.7015186447152839*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))) + 0.46194713364437323*gelu(0.8787527865412903 + -0.5088515809171308*gelu(0.8128842995226302 + -0.6732123243787744*$(x[1]) + 0.1084522267603889*$(x[2])) + 0.6075654882064532*gelu(0.14948350361517537 + 0.13810367040007687*$(x[1]) + -0.8811357247877387*$(x[2])) + -0.15666078264515404*gelu(-0.12306593354975703 + 0.895637456476829*$(x[1]) + -0.998670291965341*$(x[2])) + 0.5247142129654931*gelu(0.39065290208359826 + -0.7416515060748665*$(x[1]) + -0.262346779676371*$(x[2]))))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    