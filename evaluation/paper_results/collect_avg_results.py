import json
import numpy as np

model_names = [
    "pixel-linguist-2-pretrain",
    "pixel-linguist-2-midtrain",
]

# 1. STS English tasks (no suffix here; .json is appended when reading)
STS_English_Tasks = [
    "STS12VisualSTS",
    "STS13VisualSTS",
    "STS14VisualSTS",
    "STS15VisualSTS",
    "STS16VisualSTS",
    "STS17MultilingualVisualSTS",    
    "STSBenchmarkMultilingualVisualSTS"
]

# 2. Visual STS files used for the non-English subsets
# Combines cross-lingual (STS17) and multilingual (STSBenchmark)
Visual_STS_Files = [
    "STS17MultilingualVisualSTS.json",
    "STSBenchmarkMultilingualVisualSTS.json"
]

# 3. ViDoRe tasks (already include the .json suffix)
ViDoRe_Tasks = [
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

# Header
print(f"{'Model Name':<20} & {'STS_English':<10} & {'Visual_STS':<10} & {'ViDoRe':<10}")
print("-" * 60)

for model_name in model_names:
    # --- STS_English mean ---
    scores_sts_english = []
    for task in STS_English_Tasks:
        task_name = task + ".json"
        try:
            with open(f"./{model_name}/{task_name}", "r") as f:
                results = json.load(f)
                testset = results["scores"]["test"]
                # Single-entry scores are taken directly; multi-entry picks eng-Latn
                if len(testset) == 1:
                    scores_sts_english.append(testset[0]["main_score"] * 100)
                else:
                    for lang in testset:
                        if lang.get("languages") == ["eng-Latn"]:
                            scores_sts_english.append(lang["main_score"] * 100)
        except Exception as e:
            # print(f"Error reading {task_name} for {model_name}: {e}")
            pass
    
    mean_sts_english = np.mean(scores_sts_english) if scores_sts_english else 0.0

    # --- Visual_STS mean (cross + multi) ---
    scores_visual_sts = []
    for task_name in Visual_STS_Files:
        try:
            with open(f"./{model_name}/{task_name}", "r") as f:
                results = json.load(f)
                testset = results["scores"]["test"]
                for lang in testset:
                    # Take every non eng-Latn score
                    if lang.get("languages") != ["eng-Latn"]:
                        scores_visual_sts.append(lang["main_score"] * 100)
        except Exception as e:
            pass
            
    mean_visual_sts = np.mean(scores_visual_sts) if scores_visual_sts else 0.0

    # --- ViDoRe mean ---
    scores_vidore = []
    for task_name in ViDoRe_Tasks:
        try:
            with open(f"./{model_name}/{task_name}", "r") as f:
                results = json.load(f)
                testset = results["scores"]["test"]
                # ViDoRe normally has a single score, but handle the multi-entry case too
                if len(testset) == 1:
                    scores_vidore.append(testset[0]["main_score"] * 100)
                else:
                    for lang in testset:
                        if lang.get("languages") == ["eng-Latn"]:
                            scores_vidore.append(lang["main_score"] * 100)
        except Exception as e:
            pass
            
    mean_vidore = np.mean(scores_vidore) if scores_vidore else 0.0

    # --- Output ---
    # Format: model name & STS_English mean & Visual_STS mean & ViDoRe mean
    print(f"{model_name} & {round(mean_sts_english, 2)} & {round(mean_visual_sts, 2)} & {round(mean_vidore, 2)}")