"""Inspect saved GPU results locally without loading a model."""
import argparse
import json
from pathlib import Path
import gradio as gr
import pandas as pd


def build_review(directory):
    prompts=json.loads((directory/'prompts.json').read_text())
    metadata=json.loads((directory/'metadata.json').read_text())
    available=[p for p in prompts if (directory/(p['id']+'.json')).exists()]
    def show(prompt_id, condition):
        record=json.loads((directory/(prompt_id+'.json')).read_text())
        conditions=record['conditions']
        if condition not in conditions:
            condition='recipe'
        original=conditions['original']
        selected=conditions[condition]
        plots=[]
        for x in metadata['calibration']:
            key=f"L{x['layer']}N{x['neuron']}"
            rows=[]
            for response,stage,label in [(original,'before','Original'),(selected,'before','Selected before'),(selected,'after','Selected after')]:
                values=response['activation_trace'][key][stage][0]
                rows.extend(dict(token_position=i,activation=v,condition=label) for i,v in enumerate(values))
            plots.append(pd.DataFrame(rows))
        notes=f"Original: {len(original['token_ids'])} tokens. Selected: {len(selected['token_ids'])} tokens. "
        notes+=f"Token limit reached: original={original['reached_token_limit']}, selected={selected['reached_token_limit']}. "
        notes+="A token limit can cut off the response. These are single-turn text outputs, not observations of an agent taking action."
        choices=[c for c in conditions if c != 'original']
        return record['prompt'],original['text'],selected['text'],*plots,notes,json.dumps(selected.get('interventions',[])),gr.Dropdown(choices=choices,value=condition)
    with gr.Blocks(title='Two-neuron experiment results') as demo:
        gr.Markdown('# Two-neuron experiment\nSaved Qwen3.5-4B GPU responses. Same weights, same prompts, greedy decoding. No model runs on this page.')
        with gr.Row():
            prompt=gr.Dropdown(choices=[(p['id']+' · '+p['category'],p['id']) for p in available],value=available[0]['id'],label='Prompt')
            condition=gr.Dropdown(choices=['recipe','L14_only','L13_only','recipe_x0.5','recipe_x-1'],value='recipe',label='Condition')
        gr.Markdown('Extra conditions were run on six fixed diagnostic prompts. The condition list shows the runs available for each prompt.')
        question=gr.Textbox(label='Exact prompt',interactive=False)
        with gr.Row():
            original=gr.Textbox(label='Original model',lines=16,interactive=False)
            changed=gr.Textbox(label='Selected intervention',lines=16,interactive=False)
        notes=gr.Textbox(label='Reading the result',interactive=False)
        with gr.Accordion('Prompt-token activation measurements',open=False):
            gr.Markdown('Signed scalar activations before the MLP down projection. Positions include the chat template. The earlier layer 13 change can affect layer 14 before its own addition. These values are not readable thoughts.')
            plot14=gr.LinePlot(x='token_position',y='activation',color='condition',title='L14 N8949')
            plot13=gr.LinePlot(x='token_position',y='activation',color='condition',title='L13 N7956')
            deltas=gr.Textbox(label='Applied [layer, neuron, additive delta]',interactive=False)
        outputs=[question,original,changed,plot14,plot13,notes,deltas,condition]
        prompt.input(show,[prompt,condition],outputs)
        condition.input(show,[prompt,condition],outputs)
        demo.load(show,[prompt,condition],outputs)
    return demo


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path(__file__).resolve().parent/'results/trial-01')
    parser.add_argument('--port',type=int,default=7861)
    args=parser.parse_args()
    build_review(args.run).launch(server_name='127.0.0.1',server_port=args.port,share=False)
