# intro


It is observed that the listwise approach and pairwise approach usually outperform the
pointwise approach. In the recent Yahoo Learning to Rank Challenge, LambdaMART, which
belongs to the listwise approach, achieved the best performance.

- GBFS only requires a ranking as opp to a heuristic
  - no guarannterr to lern optimal heur
  - comon approach for no-opt
  
- learning a rank through pairwise or listwise approach open results in better performance compared to pointwise approaches.
- ranking data is easier to get
- perhaps add the motivational example if that works empirically

# Background
## Planning BG

## ML Ranking BG

- mention three approaches
  - rich field in ML
  - different ways to solve it (e.g., rankSVM, NN)
  - we focus on classification-based only

- def (global) partial ordering
  - for all states in S
- def ranking function rho(s,s') in {-1,0,1}
- rank function properties (ref, antis, transiti)
  - can we break ties in a way that helps us and keep the above true
- classification problem


# Ranked based GBFS

- GBFS using ranking
  - ranking is enough
  - define optimal rank
    - for planning and using h-star
    - follows axioms
    - address tie breaking
  - if optimal rank is provided then opt solution is found
    - cite or proof
  
 
# Learning a rank
    
  - issue of how to use rho to order the queue
    - trivial approach: when adding s to the queue compare with all other states in the queue
      - insertion requires |Q| ranking computations
      
- we solve this by extracting a (global) partial ordering from the learned ranking function
  - this results in a single real number (neg, 0 or pos) that will be used as priority of the queue
  - can be thought as a generalization of heuristic

- In order to show how to obtain the partial ordering from a learnt rank, we first need to introduce our ranking NN
  - present direct ranker
  - don't say much about the NN feeding into it (top part of fig 2)
  
- show the math of direct ranker to partial ordering
  - proof the queue using that real number as priority is equivalent to have the queue sorted according to the rank function/partial ordering
  - thus, if direct ranker learn the optimal ranking, GBFS will find the optimal solution
  - highlight again that this number is not the same as H-star

- **if the motivational example worked**, then bring it back here
  - show the values learnt and hopefully they are different from h-star
  - show goose values with given much detail of goose


# Rank-based Heuristic Learning

- architecture
  - how to bring your own NN (goose, stripshgn, etc)
  - end to end training
  - retain the NN properties, e.g., if domain-indep then ranker is also domain-indep
  
  
- getting extra data
  - all the NN considered are trained using optimal plans
  - expensive but good for regression
  - we can extract more from optimal plans for the end-to-end training
  - show how
  - prove that it is a valid partial ordering

- batch evaluation
- all other consideration








# Experiments

## NNs considered
- goose
  - original goose
  - goose with new training pipeline
- stripshgn
- planning graph features from Leslie&Garret **if available**
  - extracted features --> Feedforward --> direct ranker

## Baselines
- GBFS + h-ff
- RankSVM from Leslie&Garret **if available**
- munnin **if using IPC learning-track 2023**

## Datasets

- Leaning IPC 23
- brief discussion of domains considered

## Methodology
- training procedure
- evaluation procedure
  -- fd
- cut off time and memory
- other details (GPUs, fast downward, etc)

## Results

- Does adding a ranking layer help?
  - compare NN-X vs Direct-Ranker(NN-X)

- How does it compare with baselines/state-of-the-art



# Conclusion & Related Work

- Related work
  - Rank for planning: Leslie and Garret (only one?)
      - rankSVM, hand built features based on the planning graph
      - domain specific
      - does not generalize well to larger problems (see their experiments)
  - Edelkamp paper on NeurIPS/Arxiv
    - ??
  - Results on GBFS for ranking?



