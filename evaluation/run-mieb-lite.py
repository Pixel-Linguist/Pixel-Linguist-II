import os

import mteb

from pixel_linguist2 import get_model

# The 17 tasks reported in the paper: 5 English Visual STS tasks, plus
# STS17/STSBenchmark whose non-English subsets give the cross-lingual and
# multilingual tables, plus the 10 ViDoRe subsets of MIEB-lite.
tasks = mteb.get_tasks(
    tasks=[
    # Visual STS
    "STS12VisualSTS",
    "STS13VisualSTS",
    "STS14VisualSTS",
    "STS15VisualSTS",
    "STS16VisualSTS",

    # Visual STS cross-lingual & multilingual
    "STS17MultilingualVisualSTS",
    "STSBenchmarkMultilingualVisualSTS",

    # Visual Document Retrieval (ViDoRe)
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VidoreSyntheticDocQAEnergyRetrieval",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    ]
)

# Optionally redirect datasets to a local mirror. Unset -> load from the Hub.
LOCAL_DATA_ROOT = os.environ.get("MIEB_DATA_ROOT")
if LOCAL_DATA_ROOT:
    for task in tasks:
        path = task.metadata.dataset["path"]
        if not os.path.isabs(path):
            task.metadata.dataset["path"] = os.path.join(
                LOCAL_DATA_ROOT, path.split("/")[-1]
            )

for model_name in [
    "pixel-linguist-2-pretrain",
    "pixel-linguist-2-midtrain",
]:

    model = get_model(model_name)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="./mieb-results", batch_size=8)
