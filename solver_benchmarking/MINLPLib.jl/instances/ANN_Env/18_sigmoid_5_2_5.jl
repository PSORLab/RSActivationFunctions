using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -31.730360167469108 <= q <= 25.84895899492546)

                     add_NL_constraint(m, :(sigmoid(0.0048766473277672695 + 0.6370542772950833*sigmoid(0.25009028296342795 + -0.2547820908673408*$(x[1]) + -0.7856561704813507*$(x[2]) + -0.058708649178761174*$(x[3]) + -0.7608390940355938*$(x[4]) + -0.05318346349824532*$(x[5])) + 0.7379502189935647*sigmoid(-0.9382943152388461 + 0.5207156183879813*$(x[1]) + -0.2471020507819235*$(x[2]) + -0.7557614646986384*$(x[3]) + -0.33572241425474436*$(x[4]) + -0.6239060401766734*$(x[5])) + -0.017781498731094914*sigmoid(-0.885143874020959 + 0.9842140185190744*$(x[1]) + 0.788813814030803*$(x[2]) + -0.15763218219342123*$(x[3]) + -0.7069686241723065*$(x[4]) + -0.5626424821908671*$(x[5])) + -0.30964053180650675*sigmoid(0.6131135785432997 + 0.10922488949962084*$(x[1]) + -0.12917190851327387*$(x[2]) + -0.1680916348200565*$(x[3]) + -0.8331067755239463*$(x[4]) + 0.6030847772122314*$(x[5])) + -0.19693223852539354*sigmoid(-0.16753936699968897 + -0.5793989304507368*$(x[1]) + 0.5784575652786188*$(x[2]) + 0.7312888898953878*$(x[3]) + 0.9375360522038214*$(x[4]) + -0.8845975912485904*$(x[5]))) + sigmoid(0.6750746912592471 + -0.6689814871888631*sigmoid(0.25009028296342795 + -0.2547820908673408*$(x[1]) + -0.7856561704813507*$(x[2]) + -0.058708649178761174*$(x[3]) + -0.7608390940355938*$(x[4]) + -0.05318346349824532*$(x[5])) + 0.08174772592079638*sigmoid(-0.9382943152388461 + 0.5207156183879813*$(x[1]) + -0.2471020507819235*$(x[2]) + -0.7557614646986384*$(x[3]) + -0.33572241425474436*$(x[4]) + -0.6239060401766734*$(x[5])) + 0.43040165086652493*sigmoid(-0.885143874020959 + 0.9842140185190744*$(x[1]) + 0.788813814030803*$(x[2]) + -0.15763218219342123*$(x[3]) + -0.7069686241723065*$(x[4]) + -0.5626424821908671*$(x[5])) + 0.6360324545158482*sigmoid(0.6131135785432997 + 0.10922488949962084*$(x[1]) + -0.12917190851327387*$(x[2]) + -0.1680916348200565*$(x[3]) + -0.8331067755239463*$(x[4]) + 0.6030847772122314*$(x[5])) + 0.2590621611090005*sigmoid(-0.16753936699968897 + -0.5793989304507368*$(x[1]) + 0.5784575652786188*$(x[2]) + 0.7312888898953878*$(x[3]) + 0.9375360522038214*$(x[4]) + -0.8845975912485904*$(x[5]))) + sigmoid(-0.5603004507305789 + -0.6080760485894823*sigmoid(0.25009028296342795 + -0.2547820908673408*$(x[1]) + -0.7856561704813507*$(x[2]) + -0.058708649178761174*$(x[3]) + -0.7608390940355938*$(x[4]) + -0.05318346349824532*$(x[5])) + -0.17827205367530619*sigmoid(-0.9382943152388461 + 0.5207156183879813*$(x[1]) + -0.2471020507819235*$(x[2]) + -0.7557614646986384*$(x[3]) + -0.33572241425474436*$(x[4]) + -0.6239060401766734*$(x[5])) + -0.3024929990799108*sigmoid(-0.885143874020959 + 0.9842140185190744*$(x[1]) + 0.788813814030803*$(x[2]) + -0.15763218219342123*$(x[3]) + -0.7069686241723065*$(x[4]) + -0.5626424821908671*$(x[5])) + 0.2876553916986979*sigmoid(0.6131135785432997 + 0.10922488949962084*$(x[1]) + -0.12917190851327387*$(x[2]) + -0.1680916348200565*$(x[3]) + -0.8331067755239463*$(x[4]) + 0.6030847772122314*$(x[5])) + 0.1394466763243627*sigmoid(-0.16753936699968897 + -0.5793989304507368*$(x[1]) + 0.5784575652786188*$(x[2]) + 0.7312888898953878*$(x[3]) + 0.9375360522038214*$(x[4]) + -0.8845975912485904*$(x[5]))) + sigmoid(-0.3104079613918116 + -0.8967757209559282*sigmoid(0.25009028296342795 + -0.2547820908673408*$(x[1]) + -0.7856561704813507*$(x[2]) + -0.058708649178761174*$(x[3]) + -0.7608390940355938*$(x[4]) + -0.05318346349824532*$(x[5])) + 0.9554659562201504*sigmoid(-0.9382943152388461 + 0.5207156183879813*$(x[1]) + -0.2471020507819235*$(x[2]) + -0.7557614646986384*$(x[3]) + -0.33572241425474436*$(x[4]) + -0.6239060401766734*$(x[5])) + 0.4284452595185697*sigmoid(-0.885143874020959 + 0.9842140185190744*$(x[1]) + 0.788813814030803*$(x[2]) + -0.15763218219342123*$(x[3]) + -0.7069686241723065*$(x[4]) + -0.5626424821908671*$(x[5])) + -0.8546417933790886*sigmoid(0.6131135785432997 + 0.10922488949962084*$(x[1]) + -0.12917190851327387*$(x[2]) + -0.1680916348200565*$(x[3]) + -0.8331067755239463*$(x[4]) + 0.6030847772122314*$(x[5])) + -0.9550224035691546*sigmoid(-0.16753936699968897 + -0.5793989304507368*$(x[1]) + 0.5784575652786188*$(x[2]) + 0.7312888898953878*$(x[3]) + 0.9375360522038214*$(x[4]) + -0.8845975912485904*$(x[5]))) + sigmoid(0.11524778075403264 + -0.08842997013200282*sigmoid(0.25009028296342795 + -0.2547820908673408*$(x[1]) + -0.7856561704813507*$(x[2]) + -0.058708649178761174*$(x[3]) + -0.7608390940355938*$(x[4]) + -0.05318346349824532*$(x[5])) + -0.2446780712327863*sigmoid(-0.9382943152388461 + 0.5207156183879813*$(x[1]) + -0.2471020507819235*$(x[2]) + -0.7557614646986384*$(x[3]) + -0.33572241425474436*$(x[4]) + -0.6239060401766734*$(x[5])) + 0.5457935761049009*sigmoid(-0.885143874020959 + 0.9842140185190744*$(x[1]) + 0.788813814030803*$(x[2]) + -0.15763218219342123*$(x[3]) + -0.7069686241723065*$(x[4]) + -0.5626424821908671*$(x[5])) + -0.1673283141057702*sigmoid(0.6131135785432997 + 0.10922488949962084*$(x[1]) + -0.12917190851327387*$(x[2]) + -0.1680916348200565*$(x[3]) + -0.8331067755239463*$(x[4]) + 0.6030847772122314*$(x[5])) + 0.6343389483483679*sigmoid(-0.16753936699968897 + -0.5793989304507368*$(x[1]) + 0.5784575652786188*$(x[2]) + 0.7312888898953878*$(x[3]) + 0.9375360522038214*$(x[4]) + -0.8845975912485904*$(x[5]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    