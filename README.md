# MediLLama1: Synthetic Data Generation and LLM Fine-tuning for Indian Healthcare Reasoning

**Author:** Deeptanshu Kumar  
- [Hugging Face](https://huggingface.co/Deeptanshu13)  
- [Github](https://github.com/deeptanshukumar/)

---

## Project Overview

This project demonstrates how to generate synthetic data and fine-tune a small Large Language Model (LLM) for a healthcare consultation assistant tailored to the Indian context. The aim is to counter Western data bias present in most health models and provide more culturally and medically relevant support for Indian users.

### Key Steps:
1. **Synthetic Data Creation:** Building an Indian healthcare dataset using real and LLM-generated content.
2. **Fine-tuning:** Adapting a small LLM (e.g., Llama 3.2 1B) to this data.
3. **Evaluation:** Comparing model performance before and after fine-tuning.
4. **Resource Efficiency:** Designed to run on cloud GPUs like the NVIDIA T4.

---

## Problem Statement

**Goal:**  
Build an AI-powered Healthcare Reasoning/Consultation Assistant for India.  
- **Why?** Most open-source health LLMs are trained on Western data and fail to capture the unique disease prevalence and healthcare context in India and similar regions.

- **What?** The assistant takes a natural language symptom query and outputs:
  - Likely disease
  - Precautions
  - Rationale
  - Advice on whether to seek professional help
  - Indian context notes (e.g., rural/urban factors, local names)

---

## Synthetic Data Generation

- **Data Source:**  
  - Kaggle dataset: [Disease-Symptom-Description](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

- **Processing Pipeline:**
  1. Load and merge multiple CSVs (symptoms, descriptions, precautions, severity).
  2. Calculate per-sample severity scores using symptom weights.
  3. Balance dataset: 41 diseases × 15 samples = 615 base rows.
  4. Generate:
     - Rationale: Why was this disease recommended?
     - Indian Context Notes: Add regional details using an LLM.
     - Doctor Consultation Recommendation: Rule-based (critical diseases always recommend seeing a doctor).
  5. Q&A Formatting: Convert structured data to patient/doctor conversational pairs using a small LLM (Qwen 0.5B).
  6. Paraphrasing: Double the dataset size with LLM-based paraphrases.

- **Final Dataset:**  
  - 1,230 rows
  - Key columns for fine-tuning: `input_text` and `output_text`
  - [See Hugging Face Dataset](https://huggingface.co/datasets/Deeptanshu13/synthetic_healthcare_data_qa_pairs)

---

## Model Fine-tuning

- **Model:**  
  - [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) (or similar small, open models)
  - PEFT/LoRA: Parameter-efficient fine-tuning for fast, memory-light training.

- **Training Approach:**
  - Format:  
    ```
    ### Input:
    {input_text}

    ### Output:
    {output_text}
    ```
  - Train/Eval split: 90% / 10%
  - LoRA adapters: Only ~0.9% of parameters are trainable.

- **Hardware:**  
  - Designed to run on a single T4 GPU in Google Colab.

---

## Evaluation

- **Automatic Metrics:**  
  - ROUGE (measures string overlap between predicted and target responses).
  - Fine-tuned model shows major improvement over the base model.

- **Qualitative Analysis:**  
  - Side-by-side comparison of model responses to realistic patient queries.
  - Fine-tuned model generates Indian-relevant, structured, and actionable responses.

---

## Results

| Metric      | Base Model | Fine-Tuned Model | Improvement |
|-------------|------------|------------------|-------------|
| ROUGE-1     | ~0.07      | ~0.38            | +0.31       |
| ROUGE-L     | ~0.06      | ~0.33            | +0.27       |

- **Fine-tuned model** reliably produces disease predictions, precautions, rationale, and Indian context.
- **Base model** fails to follow the structured response or provide useful content.

---

## Key Insights & Limitations

- **Synthetic Data:** Can inject local knowledge into LLMs even when real data is scarce.
- **Small LLMs + PEFT:** Practical for resource-constrained settings.
- **Limitations:**  
  - Synthetic context quality depends on the generator LLM.
  - Medical accuracy not guaranteed (not for real clinical use).
  - Only English language; could be improved with local language data.

---

## How to Use

1. **Dataset:**  
   - Download from [Hugging Face](https://huggingface.co/datasets/Deeptanshu13/synthetic_healthcare_data_qa_pairs)

2. **Model Training:**  
   - Run the notebook (`MediLLama1.ipynb`)
   - Install requirements: `transformers`, `peft`, `trl`, `datasets`, `evaluate`, `bitsandbytes`, etc.
   - Set up Hugging Face authentication.
   - Train using the prescribed schema.

3. **Model Inference:**  
   - Use the fine-tuned model to answer new symptom queries.

---

## References

- [PEFT documentation](https://huggingface.co/docs/peft/tutorial/peft_model_config)
- [LoRA documentation](https://huggingface.co/docs/peft/task_guides/lora_based_methods)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt)
- [Kaggle Disease-Symptom Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [TRl documentation](https://huggingface.co/docs/trl/quickstart)
- [ROUGE metric documentation](https://huggingface.co/spaces/evaluate-metric/rouge)

---

*This notebook was created as part of a Research Internship assignment. For any questions or suggestions, please contact [Deeptanshu Kumar](https://github.com/deeptanshukumar/).*
