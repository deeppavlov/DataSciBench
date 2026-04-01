import os
import sys
from modulefinder import ModuleFinder

finder = ModuleFinder(path=[
    ".",
    "MetaGPT",
    "/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench",
    "/home/kaneki/Documents/DeepPavlov/подбор бенчмарка/DataSciBench/MetaGPT"
] + sys.path)

entry_points = [
    "experiments/run_examples.py",
    "experiments/evaluate.py",
    "evaluations/check_result.py",
    "role/sci_data_interpreter.py",
    "src/evaluator/cr_evaluator.py"
]

for script in entry_points:
    if os.path.exists(script):
        finder.run_script(script)

used_files = set()
for _name, mod in finder.modules.items():
    if mod.__file__ and "MetaGPT/metagpt" in mod.__file__:
        used_files.add(os.path.abspath(mod.__file__))

# All files in MetaGPT/metagpt
all_files = set()
for root, _dirs, files in os.walk("MetaGPT/metagpt"):
    for file in files:
        if file.endswith(".py"):
            all_files.add(os.path.abspath(os.path.join(root, file)))

unused = all_files - used_files

print(f"Total MetaGPT files: {len(all_files)}")
print(f"Used MetaGPT files: {len(used_files)}")
print(f"Unused MetaGPT files: {len(unused)}")

with open("unused_metagpt.txt", "w") as f:
    for file in sorted(unused):
        f.write(file + "\n")
