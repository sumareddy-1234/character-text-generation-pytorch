\# Character-Level Text Generation Comparison Report



\## Overview



This project compares two sequence models for character-level text generation using PyTorch:



\- LSTM (Long Short-Term Memory)

\- Transformer



Both models were trained on the Tiny Shakespeare dataset and evaluated using loss, perplexity, and qualitative text generation.



\---



\### Perplexity Comparison



Perplexity was calculated using:



Perplexity = exp(loss)



| Model | Final Loss | Approximate Perplexity |

|-------|------------|------------------------|

| LSTM | 1.4465 | 4.25 |

| Transformer | 0.0633 | 1.07 |



\### Analysis



The Transformer achieved a much lower perplexity score compared to the LSTM, indicating that it became highly confident in predicting the next character.



However, qualitative analysis revealed that the Transformer overfitted the dataset and generated repetitive outputs such as repeated characters and patterns.



The LSTM achieved a higher perplexity score but produced more readable and natural-looking text samples.



\---



\### Qualitative Analysis



\#### LSTM Observations



The LSTM generated text that partially resembled Shakespearean writing style. It learned:



\- word spacing

\- punctuation

\- sentence-like structures

\- character relationships



At lower temperatures, the output became more coherent and stable.



At higher temperatures, the output became more creative but less meaningful.



Example:



> "To be or not to be to the lave be platest the to seether..."



This shows that the LSTM captured some language structure even though the text was imperfect.



\---



\#### Transformer Observations



The Transformer learned patterns very quickly and achieved extremely low training loss.



However, generated outputs became repetitive:



> "To be or not to beeeeeeeeeeeeeeeee..."



This indicates overfitting and reduced diversity in generation.



Although the Transformer performed better numerically, the generated text quality was less natural compared to the LSTM.



\---



\### Temperature Analysis



Three temperature values were tested:



| Temperature | Behavior |

|-------------|----------|

| 0.5 | More deterministic and repetitive |

| 1.0 | Balanced generation |

| 1.5 | More random and creative |



For the LSTM:

\- lower temperature improved readability

\- higher temperature increased randomness



For the Transformer:

\- repetitive behavior remained even at different temperatures due to overfitting



\---



\### Challenges Faced



1\. Long CPU training time

2\. Initial generation errors caused by incorrect dictionary key access

3\. Loss curve image initially appeared blank because only one epoch was trained

4\. Transformer overfitting caused repetitive outputs

5\. Git ignored PNG files because of `.gitignore`



\---



\### Solutions Implemented



1\. Reduced model complexity and trained incrementally

2\. Fixed dictionary key mismatch in `generate.py`

3\. Trained both models for 5 epochs to generate meaningful curves

4\. Compared qualitative outputs instead of relying only on loss values

5\. Used Git force add and updated `.gitignore` settings when necessary



\---



\## Final Conclusion



The project successfully implemented and compared LSTM and Transformer models for character-level text generation using PyTorch.



The LSTM produced more human-like and diverse text, while the Transformer achieved lower perplexity but suffered from overfitting and repetitive generation.



This project provided practical understanding of:

\- sequence modeling

\- recurrent neural networks

\- transformers

\- temperature sampling

\- text generation

\- PyTorch training workflows

\- model evaluation techniques

