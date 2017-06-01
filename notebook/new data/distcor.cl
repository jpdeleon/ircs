procedure distcor(inlist, outlist, database)
	  file inlist   {prompt='List of flat corrected images with @mark'}
	  file outlist  {prompt='List of output distortion corrected images with @mark'}
	  file database {prompt='Distortion solution in the IRAF geomap database format'}
begin
	char _inlist, _outlist, _database
	char p1,p2
	char gmp

	_inlist = inlist
	_outlist = outlist
	_database = database

	list = _database
	gmp = 'none'
	while(fscan(list, p1, p2) != EOF){
		if (p1 == 'begin'){
		   gmp = p2
		}
	}

	if (gmp == 'none'){
	   print('no coodinate transforms found in the database')
	   exit
	} 
	
	unlearn('geotran')
	geotran(inlist, outlist, database, gmp)


end

