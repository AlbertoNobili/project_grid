#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "qlearn.h"

#define MAXSTA	100		    // max # of states
#define MAXACT	100		    // max # of actions
#define MAXEPI  1000000     // max # of epics
int goalstate;
static int T[MAXSTA][MAXACT];	// transition matrix
static int R[MAXSTA][MAXACT];	// reward matrix	

void init_Tmatrix()
{
    T[0][0] = 0;
    T[0][1] = 1;
    T[0][2] = 4;
    T[0][3] = 0;
    T[1][0] = 1;
    T[1][1] = 2;
    T[1][2] = 5;
    T[1][3] = 0;
    T[2][0] = 2;
    T[2][1] = 3;
    T[2][2] = 6;
    T[2][3] = 1;
    T[3][0] = 3;
    T[3][1] = 3;
    T[3][2] = 7;
    T[3][3] = 2;
    T[4][0] = 0;
    T[4][1] = 5;
    T[4][2] = 4;
    T[4][3] = 4;
    T[5][0] = 1;
    T[5][1] = 6;
    T[5][2] = 5;
    T[5][3] = 4;
    T[6][0] = 2;
    T[6][1] = 7;
    T[6][2] = 6;
    T[6][3] = 5;
    T[7][0] = 3;
    T[7][1] = 7;
    T[7][2] = 7;
    T[7][3] = 6;
}

void init_Rmatrix()
{
    R[0][0] = -2;
    R[0][1] = -1;
    R[0][2] = -1;
    R[0][3] = -2;
    R[1][0] = -2;
    R[1][1] = -1;
    R[1][2] = -5;
    R[1][3] = -1;
    R[2][0] = -2;
    R[2][1] = -5;
    R[2][2] = -1;
    R[2][3] = -1;
    R[3][0] = -2;
    R[3][1] = -2;
    R[3][2] = 10;
    R[3][3] = -1;
    R[4][0] = -1;
    R[4][1] = -5;
    R[4][2] = -2;
    R[4][3] = -2;
    R[5][0] = -1;
    R[5][1] = -1;
    R[5][2] = -2;
    R[5][3] = -1;
    R[6][0] = -1;
    R[6][1] = 10;
    R[6][2] = -2;
    R[6][2] = -5;
    R[7][0] = -5;
    R[7][1] = -2;
    R[7][2] = -2;
    R[7][3] = -1;
}

// Learning cycle within an episode
float learn_episode (int s0)
{
int s, a, r, snew;
float newerr;
float err = 0;  // average TD error
int step = 0;

    s = s0;
    while (s != goalstate){
        step++;
        a = ql_egreedy_policy(s);
        snew = T[s][a];
        r = R[s][a];
        newerr = ql_updateQ(s, a, r, snew);
        err +=  (newerr - err)/step;
        s = snew;
        //printf("finito ciclo %d\n", step);
    }
    return err;
}

// Full learning loop
float qlearn()
{
float err;      // average TD error over an episode
int s0;         // initial state;
int epic = 0;   // episode counter
    
    do{
        epic++;
        s0 = rand()%8;
        //display_environment();
        err = learn_episode(s0);
        //display_error(epic, err);
        ql_reduce_exploration();
        //printf("Fine epoca %d\n", epic);
        //l_print_Qmatrix();
    } while (epic < MAXEPI);
    return err;
}

int main()
{
    srand(time(NULL));
    init_Tmatrix();
    init_Rmatrix();
    goalstate = 7;
    //start_grapichs();
    ql_init(8, 4);  //8 states, 4 actions
    ql_set_learning_rate(0.5);
    ql_set_discount_factor(0.9);
    ql_set_expl_range(1.0, 0.1);
    ql_set_expl_decay(0.95);
    //display_menu();
    //display_param();
    //display_environment();
    //interpreter();
    qlearn();
    ql_print_Qmatrix();
    //end_graphics();
    return 0;
}