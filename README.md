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
    * preprocess_dataset
    * sft
    * train_sentiment_classifier
    * reward_model
    * ppo_rlhf
    * run_ppo
    * compare_model and plot

  * requirements.txt
  * readme.md

### instruction
1. install requirements 
2. run model.py to download model
3. run preprocess_dataset to download and tokenize dataset
4. run sft baseline to train and evaluate
5. pretrain the sentiment classfier to get the reward model
6. run ppo
7. evalute and plot