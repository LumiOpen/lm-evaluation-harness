# LumiOpen evaluations
## Math-500
We provide our own implementation of Math-500. This combines elements from the hendrycks_math500 included in lm-eval-harness and the Math-500 implementation from lighteval. The main motivation for this is that the lighteval implementation is better suited for instruction and reasoning tuned models. In comparison to the hendrycks_math500, we utilize prompts tailored for instruction tuned models and improve the parsing and normalization of the generated answer.

We also provide a machine translated Finnish version of Math-500.

### Math-500 implementation comparison
As the implementations have drastic differences, the scores also contain some variance.
Some numbers for comparison are listed below.

| Model                    | Lighteval Math-500 | hendrycks_math500 | lumiopen_math500 |
|--------------------------|--------------------|-------------------|------------------|
| Llama-Poro-2-8B-Instruct | 0.47               | 0                 | 0.444            |
| Qwen/Qwen3-8B            | 0.956              | 0                 | 0.888            |