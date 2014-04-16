#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <math.h>

//States = positions on 2D grid
//Actions = motion on the grid (Up=0, Down=1, Left=2, Right=3)
//Transition probabilities

#ifndef max
  #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
  #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

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
    case UP:    return (rowCurr+1 == rowNext && colCurr == colNext) ? 1.0 : 0.0;
    case DOWN:  return (rowCurr-1 == rowNext && colCurr == colNext) ? 1.0 : 0.0;
    case LEFT:  return (rowCurr == rowNext && colCurr-1 == colNext) ? 1.0 : 0.0;
    case RIGHT: return (rowCurr == rowNext && colCurr+1 == colNext) ? 1.0 : 0.0;
  }
  printf("Ruh roh...\n");
  return 0;
}

void compute_mdp_values(double* input, int inputLen, double* output, int outputRows, int outputCols) {
  int NUM_GRID_ENTRIES = outputRows * outputCols;

  const double STOP_RESID = 0.001;
  const int MAX_ITERS = 100;//10000;

  double currMaxResidual = DBL_MAX;

  //Note: Actions are "move directions" and are synonymous with the set of possible next states
  //from a given state S. ie: If I attempt to take action A in state S, then the next state S'
  //will result from actually executing that action A or some other one (by mistake) with some probability.

  //Discount gamma
  int iOff = 0;
  double gamma = input[iOff++];

  //Init the output values to 0
  double* V = output;
  for (int i =  0; i < NUM_GRID_ENTRIES; i++)
    V[i] = 0;

  //Init the reward values over the grid
  assert((inputLen-1) % 3 == 0);
  const int VALS_PER_REWARD = 3;
  int numRewards = (inputLen-1) / VALS_PER_REWARD;
  double* R = malloc(NUM_GRID_ENTRIES*sizeof(double));
  for (int i = 0; i < NUM_GRID_ENTRIES; i++)
    R[i] = 0;
  for (int rewardIdx = 0; rewardIdx < numRewards; rewardIdx++) {
    int rewardRow = (outputRows-1) * input[iOff + 0];
    int rewardCol = (outputCols-1) * input[iOff + 1];
    double reward = input[iOff + 2];
    iOff += VALS_PER_REWARD;
    R[GIDX(rewardRow, rewardCol, outputCols)] = reward;
    // printf("Reward set: (%d, %d), %f\n", rewardRow, rewardCol, reward);
  }

  int currIter = 0;
  while (currMaxResidual > STOP_RESID && currIter < MAX_ITERS) {
    currMaxResidual = 0;
    //Iterate over all grid states
    for (int row = 0; row < outputRows; row++) {
      for (int col = 0; col < outputCols; col++) {
        //Iterate over direction motion actions
        double maxQ = 0;
        for (int d = 0; d < NUM_DIRS; d++) {
          double Q = R[GIDX(row, col, outputCols)];
          //Iterate over possible next states (synonymous w/ actions) when attempting to take this action
          for (int j = 0; j < NUM_DIRS; j++) {
            int rowDelta = 0; int colDelta = 0;
            if (j == UP)          {rowDelta = -1; colDelta =  0;}
            else if (j == DOWN)   {rowDelta =  1; colDelta =  0;}
            else if (j == LEFT)   {rowDelta =  0; colDelta = -1;}
            else if (j == RIGHT)  {rowDelta =  0; colDelta =  1;}
            int rowNext = max(min(outputRows-1, row+rowDelta), 0);
            int colNext = max(min(outputRows-1, col+colDelta), 0);
            double QDelta = gamma * V[GIDX(rowNext, colNext, outputCols)] * transition_probability(row, col, rowNext, colNext, d);
            // printf("(%d, %d) -> (%d, %d)?  ", row, col, rowNext, colNext);
            // printf("Qdelta: %f\n", QDelta);
            Q += QDelta;
          }
          if (Q > maxQ)
            maxQ = Q;
        }
        //Update value
        int vIdx = GIDX(row, col, outputCols);
        double oldV = V[vIdx];
        V[vIdx] = maxQ;
        //Check how big the residual is for this incremental update
        double resid = fabs(V[vIdx] - oldV);
        // printf("Old V: %f; New V: %f; Resid: %f\n\n", oldV, maxQ, resid);
        if (resid > currMaxResidual)
          currMaxResidual = resid;
      }
    }
    // if (currIter % 10 == 0)
    //   printf("MAX RESIDUAL AT ITER %d: %f\n", currIter, currMaxResidual);
    currIter++;
  }

  // printf("FINAL RESIDUAL AT ITER %d: %f\n", currIter, currMaxResidual);

  free(R);
}