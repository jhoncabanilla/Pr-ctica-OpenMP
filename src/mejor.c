/*
 * GRUPO: 01
 * Jhon Steeven Cabanilla Alvarado
 * Miguel Chaveinte García
 * -------------------------------------------------------
 * Probabilistic approach to locate maximum heights
 * Hill Climbing + Montecarlo
 *
 * OpenMP version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2021/2022
 *
 * v1.0
 *
 * (c) 2022 Arturo Gonzalez Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<sys/time.h>
#include<omp.h>

#define	PRECISION	10000

/* 
 * Structure to represent a climbing searcher 
 * 	This structure can be changed and/or optimized by the students
 */
typedef struct {
	int id;				// Searcher identifier
	int pos_row, pos_col;		// Position in the grid
	int steps;			// Steps count
	int follows;			// When it finds an explored trail, who searched that trail
} Searcher;


/* 
 * Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/* 
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be changed and/or optimized by the students
 */
#define accessMat( arr, exp1, exp2 )	arr[ (int)(exp1) * columns + (int)(exp2) ]



/*
 * Function: Generate height for a given position
 * 	This function can be changed and/or optimized by the students
 */
int get_height( int x, int y, int rows, int columns, float x_min, float x_max, float y_min, float y_max  ) {
	/* Calculate the coordinates of the point in the ranges */
	float x_coord = x_min + ( (x_max - x_min) / rows ) * x;
	float y_coord = y_min + ( (y_max - y_min) / columns ) * y;
	/* Compute function value */
	float value = 2 * sin(x_coord) * cos(y_coord/2) + x + log( fabs(y_coord - M_PI_2) );
	/* Transform to fixed point precision */
	int fixed_point = (int)( PRECISION * value );
	return fixed_point;
}

/*
 * Function: Climbing step
 * 	This function can be changed and/or optimized by the students
 */
int climbing_step( int rows, int columns, Searcher *searchers, int search, int *heights, int *trails,  float x_min, float x_max, float y_min, float y_max ) {
int search_flag = 0;

	/* Annotate one step more, landing counts as the first step */
	searchers[ search ].steps ++;

	/* Get starting position */
	int pos_row = searchers[ search ].pos_row;
	int pos_col = searchers[ search ].pos_col;

	/* Stop if searcher finds another trail */
	int check;
	#pragma omp atomic capture
	{
	check = accessMat( trails, pos_row, pos_col ); 
	accessMat( trails, pos_row, pos_col ) = search;
	}


	if ( check != -1 ) {
		#pragma omp atomic write
		accessMat( trails, pos_row, pos_col ) = check;
		search_flag = 1;
	}
	else {
		float local_max;

		/* Compute the height */
		local_max=accessMat( heights, pos_row, pos_col ) = get_height( pos_row, pos_col, rows, columns, x_min, x_max, y_min, y_max );

		int climbing_direction = 0;
		float altura;
		if ( pos_row > 0 ) {
			/* Compute the height in the neighbor if needed */
			altura = accessMat( heights, pos_row-1, pos_col );
			if ( altura == INT_MIN ) 
				altura=accessMat( heights, pos_row-1, pos_col ) = get_height( pos_row-1, pos_col, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( altura > local_max ) {
				climbing_direction = 1;
				local_max = altura;
			}
		}
		if ( pos_row < rows-1 ) {
			/* Compute the height in the neighbor if needed */
			altura=accessMat( heights, pos_row+1, pos_col );
			if ( altura == INT_MIN )
				altura=accessMat( heights, pos_row+1, pos_col ) = get_height( pos_row+1, pos_col, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( altura > local_max ) {
				climbing_direction = 2;
				local_max = altura;
			}
		}
		if ( pos_col > 0 ) {
			/* Compute the height in the neighbor if needed */
			altura=accessMat( heights, pos_row, pos_col-1 );
			if ( altura == INT_MIN ) 
				altura=accessMat( heights, pos_row, pos_col-1 ) = get_height( pos_row, pos_col-1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( altura> local_max ) {
				climbing_direction = 3;
				local_max = altura;
			}
		}
		if ( pos_col < columns-1 ) {
			/* Compute the height in the neighbor if needed */
			altura=accessMat( heights, pos_row, pos_col+1 );
			if ( altura == INT_MIN ) 
				altura=accessMat( heights, pos_row, pos_col+1 ) = get_height( pos_row, pos_col+1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( altura > local_max ) {
				climbing_direction = 4;
				local_max = altura;
			}
		}

		/* Stop if local maximum is reached */
		if ( climbing_direction == 0 ) {
			searchers[ search ].follows = search;
			search_flag = 1;
		}

		/* Move in the chosen direction: 0 does not change coordinates */
		
		switch( climbing_direction ) {
			case 1: pos_row--; break;
			case 2: pos_row++; break;
			case 3: pos_col--; break;
			case 4: pos_col++; break;
		}
		searchers[ search ].pos_row = pos_row;
		searchers[ search ].pos_col = pos_col;
	}

	/* Return a flag to indicate if search should stop */
	return search_flag;
}


#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_heights( int rows, int columns, int *heights ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty 
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
	printf("Heights:\n");
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( heights, i, j ) != INT_MIN ) 
				printf(" %6d", accessMat( heights, i, j ) );
			else
				printf("       ");
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n\n");
}

void print_trails( int rows, int columns, int *trails ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty 
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
	printf("Trails:\n");
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( trails, i, j ) != -1 ) 
				printf("%7d", accessMat( trails, i, j ) );
			else
				printf("       ", accessMat( trails, i, j ) );
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n\n");
}
#endif // DEBUG

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<rows> <columns> <x_min> <x_max> <y_min> <y_max> <searchers_density> <short_rnd1> <short_rnd2> <short_rnd3>\n");
	fprintf(stderr,"\n");
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j;

	// Simulation data
	int rows, columns;		// Matrix sizes
	float x_min, x_max;		// Limits of the terrain x coordinates
	float y_min, y_max;		// Limits of the terrain y coordinates

	float searchers_density;	// Density of hill climbing searchers
	unsigned short random_seq[3];	// Status of the random sequence

	int *heights;			// Heights of the terrain points
	int *trails;			// Searchers trace and trails
	//int *tainted;			// Position found in a search
	int num_searchers;		// Number of searchers
	Searcher *searchers;		// Searchers data
	int *total_steps;		// Annotate accumulated steps to local maximums


	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 11) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	rows = atoi( argv[1] );
	columns = atoi( argv[2] );
	x_min = atof( argv[3] );
	x_max = atof( argv[4] );
	y_min = atof( argv[5] );
	y_max = atof( argv[6] );
	searchers_density = atof( argv[7] );

	/* 1.3. Read random sequences initializer */
	for( i=0; i<3; i++ ) {
		random_seq[i] = (unsigned short)atoi( argv[8+i] );
	}


#ifdef DEBUG
	/* 1.4. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d\n", rows, columns);
	printf("Arguments, x_range: ( %d, %d ), y_range( %d, %d )\n", x_min, x_max, y_min, y_max );
	printf("Arguments, searchers_density: %f\n", searchers_density );
	printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", random_seq[0], random_seq[1], random_seq[2]);
	printf("\n");
#endif // DEBUG


	/* 2. Start global timer */
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */
	
	/* 3. Initialization */
	/* 3.1. Memory allocation */
	num_searchers = (int)( rows * columns * searchers_density );
	searchers = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	total_steps = (int *)malloc( sizeof(int) * num_searchers ); 
	if ( searchers == NULL || total_steps == NULL ) {
		fprintf(stderr,"-- Error allocating searchers structures for size: %d\n", num_searchers );
		exit( EXIT_FAILURE );
	}

	heights = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	trails = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	//tainted = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	if ( heights == NULL || trails == NULL  ) { //|| tainted == NULL
		fprintf(stderr,"-- Error allocating terrain structures for size: %d x %d \n", rows, columns );
		exit( EXIT_FAILURE );
	}
	

	int num_local_max = 0;
	int max_height = INT_MIN;
	int max_accum_steps = INT_MIN;
	int total_tainted = 0;
	
		
	/* 3.2. Terrain initialization */
	#pragma omp parallel for 
	for( i=0; i<rows*columns; i++ ) {
		heights[i]= INT_MIN;
		trails[i]= -1;
		//tainted[i] = 0;
	}
	
	
	/* 3.3. Searchers initialization */
	int search;
		
	for( search = 0; search < num_searchers; search++ ) {
		searchers[ search ].pos_row = (int)( rows * erand48( random_seq ) );
		searchers[ search ].pos_col = (int)( columns * erand48( random_seq ) );
	}

	#pragma omp parallel
	{ //Comienzo región paralela
	
	#pragma omp for 
	for( search = 0; search < num_searchers; search++ ) {
		searchers[ search ].id = search;
		searchers[ search ].steps = 0;
		searchers[ search ].follows = -1;
		total_steps[ search ] = 0;
	}

	
	/* 4. Compute searchers climbing trails */
	#pragma omp for private(search)
	for( search = 0; search < num_searchers; search++ ) {
		int search_flag = 0;
		while( ! search_flag ) {
			search_flag = climbing_step( rows, columns, searchers, search, heights, trails, x_min, x_max, y_min, y_max );

#ifdef DEBUG
#ifndef _OPENMP
/* 
 * This function is used only in sequential versions. Several threads exploring 
 * at the same time can derive in mixed lines and confusing output of no value.
 */
print_trails( rows, columns, trails );
print_heights( rows, columns, heights );
#endif
#endif
		}
		
	}
	#pragma omp for private(search)
	for( search = 0; search < num_searchers; search++ ) {
		searchers[ search ].follows = accessMat( trails, searchers[ search ].pos_row, searchers[ search ].pos_col );
	}



#ifdef DEBUG
/* Print computed heights at the end of the search */
print_heights( rows, columns, heights );
#endif


	/* 5. Compute the leading follower of each searcher */
	#pragma omp for private(search)
	for( search = 0; search < num_searchers; search++ ) {
		int search_flag = 0;
		int parent = search;
		int follows_to = searchers[ parent ].follows;
		
		while( ! search_flag ) {
			if ( follows_to == parent ) search_flag = 1;
			else {
				parent = follows_to;
				follows_to = searchers[ parent ].follows;
			}
		}
		searchers[ search ].follows = follows_to;
	}
	
	
	/* 6. Compute accumulated trail steps to each maximum */
	#pragma omp single
	{
	for( search = 0; search < num_searchers; search++ ) {
		int pos_max = searchers[ search ].follows;
		//#pragma omp atomic update
		total_steps[ pos_max ] += searchers[ search ].steps;
	}
	}
	
	
	/* 7. Compute statistical data */
	#pragma omp for private(search) reduction(max:max_accum_steps,max_height) reduction(+:num_local_max)
	for( search = 0; search < num_searchers; search++ ) {
		/* Maximum of accumulated trail steps to a local maximum */
		if ( max_accum_steps < total_steps[ search ] ) 
			max_accum_steps = total_steps[ search ];

		/* If this searcher found a maximum, check the maximum value */
		if ( searchers[ search ].follows == search ) {
			num_local_max++;
			int pos_row = searchers[ search ].pos_row;
			int pos_col = searchers[ search ].pos_col;
			int altura=accessMat( heights, pos_row, pos_col );
			if ( max_height <  altura) 
				max_height = altura;
		}
	}

	#pragma omp for reduction(+:total_tainted) 
	for( i=0; i<rows*columns; i++ ) {
		if ( trails[i] !=-1) 
			total_tainted++;
	}
	
	} //Fin región paralela
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	printf("\n");
	/* 6.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	/* 6.2. Results: Statistics */
	printf("Result: %d, %d, %d, %d\n\n", 
			num_local_max,
			max_height,
			max_accum_steps,
			total_tainted );
			
		
	/* 7. Free resources */	
	free( searchers );
	free( total_steps );
	free( heights );
	free( trails );
	//free( tainted );

	/* 8. End */
	return 0;
}
