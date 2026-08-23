import json
import numpy as np
model_names = [
    "pixel-linguist-2-pretrain",
    "pixel-linguist-2-midtrain",
]
STS_English = [
    "STS12VisualSTS",
    "STS13VisualSTS",
    "STS14VisualSTS",
    "STS15VisualSTS",
    "STS16VisualSTS",
    "STS17MultilingualVisualSTS",   
    "STSBenchmarkMultilingualVisualSTS"
]
    
print("Visual STS (English)")
for model_name in model_names:
    current_result = []
    for task in STS_English:
        task_name = task+".json"
        with open(f"./{model_name}/{task_name}","r") as f:
            results = json.load(f)
            testset = results["scores"]["test"]
            if len(testset) == 1:
                # print(task_name)
                current_result.append(round(testset[0]["main_score"]*100,2))
            else:
                for lang in testset:
                    if lang["languages"] == ["eng-Latn"]:
                        # print(task_name)
                        current_result.append(round(lang["main_score"]*100,2))
    current_result.append(round(np.mean(current_result),2))
    print(model_name, "&", " & ".join([str(i) for i in current_result]))


print("\n\n\n")

print("Visual STS (cross)")
for model_name in model_names:
    current_result = []
    task_name = "STS17MultilingualVisualSTS.json"
    with open(f"./{model_name}/{task_name}","r") as f:
        results = json.load(f)
        testset = results["scores"]["test"]
        for lang in testset:
            if not lang["languages"] == ["eng-Latn"]:
                current_result.append(round(lang["main_score"]*100,2))
    current_result.append(round(np.mean(current_result),2))
    print(model_name, "&", " & ".join([str(i) for i in current_result]))

print("\n\n\n")


print("Visual STS (multi)")

for model_name in model_names:
    current_result = []
    task_name = "STSBenchmarkMultilingualVisualSTS.json"
    with open(f"./{model_name}/{task_name}","r") as f:
        results = json.load(f)
        testset = results["scores"]["test"]
        for lang in testset:
            if not lang["languages"] == ["eng-Latn"]:
                current_result.append(round(lang["main_score"]*100,2))
    current_result.append(round(np.mean(current_result),2))
    print(model_name, "&", " & ".join([str(i) for i in current_result]))


print("\n\n\n")

print("ViDoRe")

model_names = [
    "pixel-linguist-2-pretrain",
    "pixel-linguist-2-midtrain",
]

vidore = [
    'VidoreArxivQARetrieval.json', 
    'VidoreDocVQARetrieval.json', 
    'VidoreInfoVQARetrieval.json', 
    'VidoreShiftProjectRetrieval.json', 
    'VidoreSyntheticDocQAAIRetrieval.json', 
    'VidoreSyntheticDocQAEnergyRetrieval.json', 
    'VidoreSyntheticDocQAGovernmentReportsRetrieval.json', 
    'VidoreSyntheticDocQAHealthcareIndustryRetrieval.json', 
    'VidoreTabfquadRetrieval.json', 
    'VidoreTatdqaRetrieval.json'
]
for model_name in model_names:
    current_result = []
    for task in vidore:
        task_name = task
        with open(f"./{model_name}/{task_name}","r") as f:
            results = json.load(f)
            testset = results["scores"]["test"]
            if len(testset) == 1:
                # print(task_name)
                current_result.append(round(testset[0]["main_score"]*100,2))
            else:
                for lang in testset:
                    if lang["languages"] == ["eng-Latn"]:
                        # print(task_name)
                        current_result.append(round(lang["main_score"]*100,2))
    current_result.append(round(np.mean(current_result),2))
    print(model_name, "&", " & ".join([str(i) for i in current_result]))