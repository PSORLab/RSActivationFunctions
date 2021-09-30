using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -8.71797166457218 <= q <= 8.662171616045406)

                     add_NL_constraint(m, :(tsoftplus(-0.3026329584771359 + 0.6648159395216804*tsoftplus(0.6425162425360083 + -0.1593740152823706*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.030757684192486145*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.020932871876955694*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.572075751636719*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + -0.5326586135406095*tsoftplus(0.7495915129893822 + -0.8645867452365321*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.5478285096896327*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.07183574354207778*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8744365845978099*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + 0.7484372899239822*tsoftplus(-0.2615717181561239 + -0.8036811394359291*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.3641340988145618*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8935897114685045*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9425723307935803*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + 0.8779555041184959*tsoftplus(0.2130601285896092 + -0.9992213951223436*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.6124737067476573*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8870530887762915*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9617434946041237*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))))) + tsoftplus(-0.44537482344196455 + 0.099138262770615*tsoftplus(0.6425162425360083 + -0.1593740152823706*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.030757684192486145*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.020932871876955694*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.572075751636719*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + -0.7602978032090011*tsoftplus(0.7495915129893822 + -0.8645867452365321*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.5478285096896327*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.07183574354207778*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8744365845978099*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + -0.4820687545107947*tsoftplus(-0.2615717181561239 + -0.8036811394359291*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.3641340988145618*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8935897114685045*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9425723307935803*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + -0.6568299978345387*tsoftplus(0.2130601285896092 + -0.9992213951223436*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.6124737067476573*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8870530887762915*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9617434946041237*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))))) + tsoftplus(-0.6459572467068777 + 0.3428981268219289*tsoftplus(0.6425162425360083 + -0.1593740152823706*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.030757684192486145*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.020932871876955694*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.572075751636719*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + -0.7163711189089201*tsoftplus(0.7495915129893822 + -0.8645867452365321*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.5478285096896327*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.07183574354207778*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8744365845978099*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + -0.10619956034829903*tsoftplus(-0.2615717181561239 + -0.8036811394359291*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.3641340988145618*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8935897114685045*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9425723307935803*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + 0.4311470338519148*tsoftplus(0.2130601285896092 + -0.9992213951223436*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.6124737067476573*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8870530887762915*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9617434946041237*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))))) + tsoftplus(0.3425433867167049 + 0.47077907182269607*tsoftplus(0.6425162425360083 + -0.1593740152823706*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.030757684192486145*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.020932871876955694*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.572075751636719*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + 0.1405414273601151*tsoftplus(0.7495915129893822 + -0.8645867452365321*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.5478285096896327*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.07183574354207778*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8744365845978099*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + -0.28054844000177104*tsoftplus(-0.2615717181561239 + -0.8036811394359291*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.3641340988145618*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8935897114685045*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9425723307935803*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2])))) + 0.5781082542132938*tsoftplus(0.2130601285896092 + -0.9992213951223436*tsoftplus(0.2810093593965126 + 0.13785356711816998*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.20217961066526602*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.46665125814773134*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + 0.00881841248046955*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + -0.6124737067476573*tsoftplus(0.2841884468215925 + 0.08415590844235421*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + -0.13172038232031014*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.9695449004916616*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.7042208250523259*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.8870530887762915*tsoftplus(0.6243910663985859 + 0.292977034071372*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.7233272171453993*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.5678172438872999*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.4444059405784282*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))) + 0.9617434946041237*tsoftplus(-0.786151848830086 + 0.08692546042943317*tsoftplus(0.24232633210949395 + -0.5940412022572001*$(x[1]) + 0.15377766977649765*$(x[2])) + 0.12424881664936693*tsoftplus(0.09643972131040579 + -0.4297788302376868*$(x[1]) + 0.42893501074496854*$(x[2])) + 0.09662086521649371*tsoftplus(-0.7164623556782979 + -0.8583319518757531*$(x[1]) + -0.2792766622962559*$(x[2])) + -0.22646671546863573*tsoftplus(0.3070601268213413 + -0.22557759164706104*$(x[1]) + 0.017922778694557717*$(x[2]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    