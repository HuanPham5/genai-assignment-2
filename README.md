# Setup instructions (Python version, pip install commands)

For windows 11, vscode With anaconda already installed. If not, install vscode and anaconda first.

Configure the default terminal in vscode to launch conda terminal everytime.

CTRL+SHIFT+P in open search bar setting vscode, search "Preferences: Open User Settings (JSON)" and paste the following into the json file:

<pre>
"terminal.integrated.profiles.windows": {
  "Anaconda PowerShell": {
    "path": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    "args": [
      "-NoExit",
      "-Command",
      "& 'C:\\ProgramData\\anaconda3\\shell\\condabin\\conda-hook.ps1'; conda activate base"
    ],
    "icon": "anaconda"
  }
},
"terminal.integrated.defaultProfile.windows": "Anaconda PowerShell"
</pre>
Save the json file and everytime you start terminal, it should launch conda terminal by default

Launch terminal inside vscode

Create virtual env by:

<pre>
conda create -n genai2 python=3.12

conda activate genai2
</pre>
Then type 

<pre>
pip install -r requirements.txt
</pre> 

to install all the required packages

# How to create ~/.soonerai.env and required keys
Click on the "New File..." icon on the left side panel in vscode, then type in ".soonerai.env"

Then open the .soonerai.env and paste:

<pre>
SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b
</pre>

Get the API key by sign up and register With OU email at https://ai.sooners.us

Once logged in, click top right at the profile icon and click on Settings

Then click on Account

Look for API keys and click show

Copy the API keys over to the ".soonerai.env" file and replace the "your_key_here" With the API keys

# How to run the chatbot

run the script by typing 

<pre>
python cifar10_classify.py
</pre>

Wait for it classify all 100 images.

# Results
Baseline: no system prompt

Accuracy over 100 images: 62.00%

Saved 1 misclassification rows to misclassifications.jsonl 


Prompt 1 uses the disguises the vlm as expert classifier With a note telling to carefully look at each feature before labeling them.

Accuracy over 100 images: 64.00%

Saved 0 misclassification rows to misclassifications.jsonl


Prompt 2, this time tells the vlm it has 10 classes to choose from.

Accuracy over 100 images: 63.00%

Saved 0 misclassification rows to misclassifications.jsonl


Prompt 3, tell it explicity that it's working With cifar10 dataset.

Accuracy over 100 images: 64.00%

Saved 0 misclassification rows to misclassifications.jsonl


Prompt 4, tell the vlm what the 10 labels are. Then tells where it got confuses in the earlier run/ points out it got wrong in the earlier runs but not too specific.

Accuracy over 100 images: 62.00%

Saved 1 misclassification rows to misclassifications.jsonl


Prompt 5, tells it explicity what type error, tell the structure of the data, give it some extra hints.

Accuracy over 100 images: 57.00%

Saved 0 misclassification rows to misclassifications.jsonl


Prompt 6, clean up prompt 5 is much cleaner instructions

Accuracy over 100 images: 63.00%

Saved 0 misclassification rows to misclassifications.jsonl


Prompt 7, give the answer in blocks form

Accuracy over 100 images: 59.00%

Saved 0 misclassification rows to misclassifications.jsonl


Prompt 8, attempts to fully disable vision by declaring the images “corrupted noise” and instructing the model to ignore them, instead returning labels purely by index lookup. Provides an explicit 1–100 range mapping for the CIFAR-10 classes and strict rules forbidding visual classification.

Accuracy over 100 images: 57.00%

Saved 0 misclassification rows to misclassifications.jsonl


Prompt 3-v2 --> works best!

Accuracy over 100 images: 67.00%

Saved 0 misclassification rows to misclassifications.jsonl


# Analysis

With the baseline, ie. no system prompt whatsoever, it the model correctly identifies clear laebls like airplanes, trucks, horses, but confuses classes such as birds vs. dogs or ships vs. trucks, and outputs one label outside the CIFAR-10 label, monkey.


With prompt 1, accuracy increases slightly to 64%, and the model still gets the easy classes right (airplanes, trucks, horses) while reducing some of the earlier confusion. Notably, it no longer produces any labels outside the CIFAR-10 set, though it still mixes up categories such as birds vs. dogs or frogs vs. deer.


With prompt 2, accuracy drops slightly to 63%, and the model still mixes up categories like birds vs. dogs, frogs vs. deer, or ships vs. trucks. It no longer outputs any out-of-vocabulary labels, but the prompt does not meaningfully reduce the core class-confusion errors seen in earlier runs.


With prompt 3, accuracy is back to 64%, With the model still nailing the easy classes (airplanes, trucks, horses) and producing no out-of-vocabulary labels. persistent confusions remain: birds → dogs, frogs → deer/cat/truck, and ships → trucks, With occasional airplane → ship/truck.


With prompt 4, accuracy drops to 62% (back to baseline), and the model again confuses similar classes (airplane → bird, ship → airplane/bird/truck, frog → dog/bird/horse, cat → dog). it also reintroduces an out-of-vocabulary label (“car” → unknown), indicating the prompt isn’t constraining outputs to CIFAR-10.


With prompt 5, accuracy drops sharply to 57% (worst so far). the model regresses on airplanes (→ bird/horse), automobiles (→ truck/airplane/dog/horse), frogs (→ bird/dog/cat/horse), and ships (→ bird/airplane/horse/automobile), while trucks/horses remain strong. no out-of-vocabulary labels appear, suggesting the prompt broadened cross-class drift instead of tightening decision boundaries.


With prompt 6, accuracy drops again to 60%, and the model shows broader drift across multiple classes. airplanes are now misclassified as birds, horses, trucks, and dogs; birds are repeatedly mapped to dogs; frogs are heavily confused With dogs, cats, and horses; and ships occasionally become horses or airplanes. no out-of-vocabulary labels appear, but the prompt clearly weakened class boundaries instead of strengthening them.


With prompt 7, accuracy slides further to 58%, showing even weaker class separation than before. airplanes are now misclassified as horses, trucks, dogs, and even automobiles; birds collapse heavily into “dog” and “horse”; frogs mostly get mapped to dogs or cats; and ships are confused With trucks, horses, and even automobiles. no out-of-vocabulary labels appear, but the prompt clearly destabilizes the model’s decisions instead of improving them.


With prompt 8 tries to bypass vision and force index-based answers, but the model ignores the instructions and keeps classifying visually, resulting in only 57 % accuracy.


Prompt 3-v2 achieves the highest score so far at 66 % accuracy. Unlike the heavier prompts, it does not try to “correct” the model’s behavior, override vision, or impose rules — it just frames the task plainly, and the model leans fully on visual recognition instead of overthinking or second-guessing labels.

# Error pattern

Across all prompts, the same confusions repeat: birds drift into dogs or horses, frogs scatter into deer/cat/dog/truck, and ships often become trucks or airplanes. “Easy” classes (airplane, truck, horse) stay stable throughout. Out-of-vocabulary labels only appear when the prompt is weak, and disappear once the label set is stated. Heavier prompts consistently make things worse—adding rules increases cross-class drift instead of fixing it—while the best result (66%) comes from the simplest prompt, which just states the task and dataset and lets the model rely on its own visual prior.