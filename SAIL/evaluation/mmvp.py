import csv
import os
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv
from tqdm import tqdm

def benchmark_model(model, benchmark_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_processor = model.vision_model.image_processor
    text_processor = model.text_model.tokenizer
    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')
    

    csv_outfile = open('output.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features', 
        'State and Condition', 'Quantity and Count', 
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader), total=150):
            qid1, qtype1, statement1 = row
        
            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row
            
            qid1, qid2 = int(qid1), int(qid2)
            
            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            input_text1 = text_processor(text1, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)
            input_text2 = text_processor(text2, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)

            imgs = img_processor([img1,img2], return_tensors="pt").to(device)   
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    logits_per_text1 = model(imgs, input_text1, text_list=[text1])['logits_per_text']
                    logits_per_text2 = model(imgs, input_text2, text_list=[text2])['logits_per_text']
                    
                    probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                    probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]
            
            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            
            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        csv_outfile.close()

    # Calculate percentage accuracies
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    return pair_accuracies


def mmvp_eval(model, text_model_name, vision_model_name, directory="evaluation/MMVP_VLM"):
    assert os.path.exists(directory), f"Directory {directory} does not exist"

    results_openai = {f'SAIL': benchmark_model(model, directory) }
    # Merge results
    results = {**results_openai}

    # Convert results to format suitable for star plot
    categories = results[list(results.keys())[0]].keys()
    data = {'Categories': list(categories)}
    for model in list(results_openai.keys()):
        data[model] = [results[model][category] for category in categories]

    print(results)
    # Average accuracy
    average_accuracy = sum(results[model][category] for model in list(results_openai.keys()) for category in categories) / (len(list(results_openai.keys())) * len(categories))
    print(f"Average accuracy: {average_accuracy:.2f}%")
    results[model]['Average'] = average_accuracy
    return results

