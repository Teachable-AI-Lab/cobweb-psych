# cobweb-psych
Cobweb worked in experiments on multiple psychological effects in human learning.

Submission to CogSci26.

This repository is designed to showcase effects of category learning in humans in Cobweb, a model of category learning that creates concepts in a probabilistic, unsupervised, incremental, piecemeal nature.

Details on all effects studied (and a rough planning doc) can be found [here](https://docs.google.com/document/d/1igiYFseYhwj7bZAVHMLXo0ZMpm6sqR6XJq38XC3tguA/edit?tab=t.0)!

## Structure of the repository

`experiments/` contains all datasets and relevant papers which they are inherited from, as well as a benchmark-runner script with command-line arguments 
`multinomial_cobweb/` contains all relevant code for testing and running our version of Cobweb, which inherits some properties from Cobweb/4V and other experimental features

## Documentation on Cobweb

Cobweb is an unsupervised, incremental, piecemeal concept formation system. The version we use is best inherited from [Cobweb/4V](https://arxiv.org/abs/2402.16933v2), and supports the following characteristics:
*   Basic-level as categorized by minimizing log-likelihood or maximizing pointwise mutual information
*   Multi-node + greedy prediction system
