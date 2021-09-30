using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -35.067082718186526 <= q <= 42.174245726281896)

                     add_NL_constraint(m, :(tsigmoid(-0.27240918979996076 + 0.6897894850172874*tsigmoid(0.43776653722071757 + 0.14680579591085507*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.5488651696345448*tsigmoid(-0.5992886370730184 + 0.21145558475313342*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.23513661876340164*tsigmoid(0.10072899303408356 + 0.6544462050090085*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.9286095826488316*tsigmoid(-0.8762246797551754 + -0.6962852009784499*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) + tsigmoid(-0.061621413659346036 + -0.8082389871746565*tsigmoid(0.43776653722071757 + 0.14680579591085507*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + 0.24794834385206954*tsigmoid(-0.5992886370730184 + 0.21145558475313342*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.8301386044845835*tsigmoid(0.10072899303408356 + 0.6544462050090085*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.01775457324771068*tsigmoid(-0.8762246797551754 + -0.6962852009784499*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) + tsigmoid(0.5769593139314138 + -0.6609851001362568*tsigmoid(0.43776653722071757 + 0.14680579591085507*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.0602190008831589*tsigmoid(-0.5992886370730184 + 0.21145558475313342*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.7859453937775993*tsigmoid(0.10072899303408356 + 0.6544462050090085*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + 0.8279800333045495*tsigmoid(-0.8762246797551754 + -0.6962852009784499*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) + tsigmoid(0.8571483115497647 + -0.179181384293289*tsigmoid(0.43776653722071757 + 0.14680579591085507*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.3115415524721241*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.813300326854324*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.058102891352175945*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.047754372827110814*tsigmoid(-0.5992886370730184 + 0.21145558475313342*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.20210858746771532*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.8718414810201667*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.12703778107223052*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + 0.23869055017757734*tsigmoid(0.10072899303408356 + 0.6544462050090085*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5521585433424212*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.6166400583439944*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + 0.7234168953546698*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2])))) + -0.7742380954721755*tsigmoid(-0.8762246797551754 + -0.6962852009784499*tsigmoid(0.8990796725905339 + 0.4164998637645212*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.9988082480124638*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.470410730186932*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + 0.37513596182220255*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.2459036750057897*tsigmoid(0.7054659745983756 + -0.21832961280148355*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.9633475144534573*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.3506136060626224*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7509175602794702*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.5449036992198066*tsigmoid(0.5287827920800199 + 0.5906724551105156*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + -0.04509490998063814*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + 0.9130421665460338*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.36218982997166593*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))) + -0.9620319304105247*tsigmoid(0.3783822455675745 + 0.13506485849462857*tsigmoid(0.10491236328139131 + -0.8972138986817386*$(x[1]) + -0.37867585237991674*$(x[2])) + 0.8607101418666123*tsigmoid(0.009145790267627518 + 0.7105535251443387*$(x[1]) + 0.7559891020133289*$(x[2])) + -0.07066531933782194*tsigmoid(0.42374379447610044 + 0.3435610777186686*$(x[1]) + 0.2575466013066374*$(x[2])) + -0.7064042955400209*tsigmoid(-0.16815299427254926 + -0.12656896108402238*$(x[1]) + 0.7832723693688344*$(x[2]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    