from gensim.models import KeyedVectors
import gradio as gr
file_path = 'tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
model = KeyedVectors.load_word2vec_format(file_path, binary=False)
def semantic_arithmetic(word1,word2,word3):
    result = model.most_similar(positive=[word2, word3], negative=[word1],topn=1)
    most_similar_word, highest_similarity = result[0]
    return most_similar_word

with gr.Blocks() as app:
    gr.Markdown("# 词语加减法") 
    with gr.Row():
        word1 = gr.Textbox(value="男人")
        minus1 = gr.Markdown("<span style='font-size: 50px;'> - </span>")
        word2 = gr.Textbox(value="女人")
        minus1 = gr.Markdown("<span style='font-size: 50px;'> = </span>")
        word3 = gr.Textbox(value='国王')
        minus1 = gr.Markdown("<span style='font-size: 50px;'> - </span>")
        output = gr.Label()
    btn = gr.Button("计算")
    btn.click(semantic_arithmetic, inputs=[word1, word2, word3], outputs=output)
    
app.launch(server_name="0.0.0.0", server_port=7860)
