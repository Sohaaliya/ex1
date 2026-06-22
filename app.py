from transformers import pipeline
import gradio as gr

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_text(prompt):
    output = generator(
        prompt,
        max_length=250,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return output[0]["generated_text"]

iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your prompt..."
    ),
    outputs="text",
    title="GPT-2 Text Generator"
)

iface.launch()