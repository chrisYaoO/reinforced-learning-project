# reinforced-learning-project

### dataset
1. yelp polarity(sentiment)

### model
1. distillgpt2
2. reward model not written


### Structure
* PJ
  * configs
  * data
    * tokenized_data
  * models
    * distilgpt2
  * results
    * sft_results
      * checkpoints
      * runs
  * src
    * model
    * preprocess
    * sft
  * requirements
  * readme.md

### instruction
1. install requirements 
2. run model.py to download model
3. run preprocess_dataset to download and tokenize dataset
4. run sft baseline to train and evaluate
   1. train not tuned and evaluation not tested
5. run command ```tensorboard --logdir results/sft_results/runs --port 6006``` and [localhost](http://localhost:6006/) to monitor runs