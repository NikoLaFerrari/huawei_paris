# Future Work
We list the possible directions to continue this research/project.

1. ```interpreter.py```:
- Need to enhance the current formulae -> use Hockney equations, research shows they are accurate until a certain point during scaling.
- Need to ensure the regression going on is not 'fake', for this I believe we need all the traces to be different from each other (different number of devices).

2. ```predictor.py```:
- Need to ensure correct scaling of Hockney equations: it is possible that we may be asked to predict beyond the point where accuracy of Hockney equations' does not hold, we need to ensure we have another formulae ready that works from this point of failure and above. 
- Need to add validation to check if the predicted values are realistic.

3. ```interface.py```:
- Need to run completely autonomously: 
	- Call ND for classification (-l []) & compute the ratios.
	- Call ND for obtaining the top-k strategies.
	- Run Predictor.predict() for each of the strategies and obtain the predicted scores/times.
This would be a complete integration between ND and regression.
