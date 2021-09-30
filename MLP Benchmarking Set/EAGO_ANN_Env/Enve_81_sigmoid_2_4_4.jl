using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -35.067082718186526 <= q <= 42.174245726281896)

                     add_NL_constraint(m, :(sigmoid(-0.27240918979996076 + 0.6897894850172874*sigmoid(0.43776653722071757 + 0.14680579591085507*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.5488651696345448*sigmoid(-0.5992886370730184 + 0.21145558475313342*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.23513661876340164*sigmoid(0.10072899303408356 + 0.6544462050090085*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.9286095826488316*sigmoid(-0.8762246797551754 + -0.6962852009784499*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) + sigmoid(-0.061621413659346036 + -0.8082389871746565*sigmoid(0.43776653722071757 + 0.14680579591085507*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + 0.24794834385206954*sigmoid(-0.5992886370730184 + 0.21145558475313342*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.8301386044845835*sigmoid(0.10072899303408356 + 0.6544462050090085*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.01775457324771068*sigmoid(-0.8762246797551754 + -0.6962852009784499*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) + sigmoid(0.5769593139314138 + -0.6609851001362568*sigmoid(0.43776653722071757 + 0.14680579591085507*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.0602190008831589*sigmoid(-0.5992886370730184 + 0.21145558475313342*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.7859453937775993*sigmoid(0.10072899303408356 + 0.6544462050090085*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + 0.8279800333045495*sigmoid(-0.8762246797551754 + -0.6962852009784499*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) + sigmoid(0.8571483115497647 + -0.179181384293289*sigmoid(0.43776653722071757 + 0.14680579591085507*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.047754372827110814*sigmoid(-0.5992886370730184 + 0.21145558475313342*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + 0.23869055017757734*sigmoid(0.10072899303408356 + 0.6544462050090085*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.7742380954721755*sigmoid(-0.8762246797551754 + -0.6962852009784499*sigmoid(0.8990796725905339 + 0.4164998637645212*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*sigmoid(0.7054659745983756 + -0.21832961280148355*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*sigmoid(0.5287827920800199 + 0.5906724551105156*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*sigmoid(0.3783822455675745 + 0.13506485849462857*sigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*sigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*sigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*sigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    