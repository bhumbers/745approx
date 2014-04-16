#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

//States = positions on 2D grid
//Actions = motion on the grid (Up=0, Down=1, Left=2, Right=3)
//Transition probabilities

enum {
  UP = 0,
  DOWN = 1,
  LEFT = 2,
  RIGHT = 3
};

static const int NUM_DIRS = 4;

#define GIDX(R, C, NCols) (R*NCols + C)

double transition_probability(int rowCurr, int colCurr, int rowNext, int colNext, int dir) {
  //assume deterministic, for now
  switch (dir) {
    case UP:    return (rowCurr == rowNext - 1 && colCurr == colNext) ? 1.0 : 0.0;
    case DOWN:  return (rowCurr == rowNext + 1 && colCurr == colNext) ? 1.0 : 0.0;
    case LEFT:  return (rowCurr == rowNext && colCurr == colNext - 1) ? 1.0 : 0.0;
    case RIGHT: return (rowCurr == rowNext && colCurr == colNext + 1) ? 1.0 : 0.0;
  }
  return 0;
}

void compute_mdp_values(double* input, int inputLen, double* output, int outputRows, int outputCols) {
  int NUM_GRID_ENTRIES = outputRows * outputCols;

  const double stoppingBellmanResidual = 0.001;
  double currMaxResidual = DBL_MAX;

  //Note: Actions are "move directions" and are synonymous with the set of possible next states
  //from a given state S. ie: If I attempt to take action A in state S, then the next state S'
  //will result from actually executing that action A or some other one (by mistake) with some probability.

  //Init the output values to 0
  double* V = output;
  for (int i =  0; i < NUM_GRID_ENTRIES; i++)
    V[i] = 0;

  //Init the reward values over the grid
  assert(inputLen % 3 == 0);
  const int VALS_PER_REWARD = 3;
  int numRewards = inputLen / VALS_PER_REWARD;
  double* R = malloc(NUM_GRID_ENTRIES*sizeof(double));
  for (int i = 0; i < NUM_GRID_ENTRIES; i++)
    R[i] = 0;
  for (int rewardIdx = 0; rewardIdx < numRewards; rewardIdx += VALS_PER_REWARD) {
    int rCol = (outputRows-1) * input[rewardIdx+0];
    int rRow = (outputCols-1) * input[rewardIdx+1];
    double reward = input[rewardIdx+2];
    R[GIDX(rCol, rRow, outputCols)] = reward;
  }

  while (currMaxResidual > stoppingBellmanResidual) {
    currMaxResidual = 0;
    //Iterate over all grid states
    for (int row = 0; row < outputRows; row++) {
      for (int col = 0; col < outputCols; col++) {
        //Iterate over direction motion actions
        double Q[NUM_DIRS];
        for (int d = 0; d < NUM_DIRS; d++) {
          Q[d] = R[GIDX(row, col, outputCols)];
          //Iterate over possible next states (synonymous w/ actions) when attempting to take this action
          for (int j = 0; j < NUM_DIRS; j++) {
            int rowDelta = 0; int colDelta = 0;
            if (j == UP) {rowDelta = -1; colDelta = 0;}
            else if (j == DOWN) {rowDelta = 1; colDelta = 0;}
            else if (j == LEFT) {rowDelta = 0; colDelta = -1;}
            else if (j == RIGHT) {rowDelta = 0; colDelta = 1;}
            Q[d] += V[GIDX(row, col, outputCols)] * transition_probability(row, col, row+rowDelta, col+colDelta, d);
          }
        }
      }
    }
  }

  free(R);
}