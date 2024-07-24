# utils module

### *class* utils.LengthSampler(min_length, max_length)

Bases: `object`

#### trim_to_word_boundary(text, length)

### utils.cal_coherence(prompts, continuations, model, tokenizer)

### utils.cal_diversity(sentence)

### utils.cal_gpt2_harmless_probabilities(prompts, continuations, model, tokenizer)

### utils.cal_gpt2_helpful_probabilities(prompts, continuations, model, tokenizer)

### utils.cal_harmless_probabilities(sentences, model, tokenizer)

### utils.cal_humor_probabilities(sentences, model, tokenizer)

### utils.cal_log_perplexity(sentences, model, tokenizer)

### utils.cal_positive_sentiment(sentences, model, tokenizer)

### utils.clean_and_trim_to_last_sentence(prompts, decoded_outputs)

### utils.compute_rep_n(sentence, n)

### utils.convert_ppo_modelname_to_huggingface_valid(ppo_model_name)

convert a ppo trained model name to ensure that it is a valid huggingface model path name

### utils.get_device()

### utils.get_model_and_tokenizer(model_name)

### utils.get_nvidia_smi_info()

### utils.get_prompts_from_Anthropic_harmless()

### utils.get_prompts_from_imdb()

### utils.get_reward(sentences, value, model=None, tokenizer=None, prompts=None, use_score=True)
