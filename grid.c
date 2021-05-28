#include <stdlib.h> 

#define MAXSTA	100		    // max # of states
#define MAXACT	100		    // max # of actions
#define MAXEPI  1000000L    // max # of epics
int goalstate;
static int T[MAXSTA][MAXACT];	// transition matrix
static int R[MAXSTA][MAXACT];	// reward matrix	

// Learning cycle within an episode
float learn_episode (int s0)
{
int s, a, r, snew;
int steps = 0;
float err = 0;  // accumulated TD error

    s = s0;
    while (s != goalstate){
        a = ql_egreedy_policy(s);
        snew = T[s][a];
        r = R[s][a];
        err += ql_updateQ(s, a, r, snew);
        s = snew;
        steps++;
    }
    return err/steps;
}

// Full learn loop
float qlearn(float eps)
{
float err;      // average TD error over an episode
int s0;         // initial state;
int epic = 0;   // episode counter
    
    do{
        epic++;
        s0 = get_initial_state();
        display_environment();
        err = learn_episode(s0);
        display_error(epic, err);
        ql_reduce_exploration();
    } while (epic < MAXEPI);
    return err;
}

int main()
{
    srand(time(NULL));
    start_grapichs();
    ql_init(8, 4);  //8 states, 4 actions
    ql_set_learning_rate(0.5);
    ql_set_discount_factor(0.9);
    ql_set_expl_range(1.0, 0.1);
    ql_set_expl_decay(0.95);
    display_menu();
    display_param();
    display_environment();
    interpreter();
    end_graphics();
    return 0;
}