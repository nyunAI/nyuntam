## Steps to setup on android device : 

<ol> 
<li> Install termux apk : <a href=https://github.com/termux/termux-app/releases/tag/v0.118.1> link </a>  </li>

**NOTE :** Termux might not work on latest version of android, so, it is advisable use android 9 (tested) 

<li> in termux, run : <code>apt upgrade && apt update</code></li>
<li> install python in termux : <ol> <li><code>pkg install tur-repo</code></li> <li><code>pkg install python3.11</code></li></ol>
<li> Install espeak using :  <code>pkg install espeak</code> </li>
</ol>

## Setup llama.cpp:
Setup llama.cpp on android using the following commands:
<ol>
<li><code>git clone https://github.com/ggerganov/llama.cpp.git</code></li>
<li><code>cd llama.cpp</code> </li>
<li><code>apt install git cmake</code> </li>
<li><code>make GGML_NO_LLAMAFILE=1</code></li>
</ol>

## Setup whisper.cpp:
Setup whisper.cpp on android using the following commands:
<ol>
<li> <code>git clone https://github.com/ggerganov/whisper.cpp.git</code></li>
<li> <code>cd whisper.cpp </code></li>
<li> <code>make</code> </li>
</ol>

## Getting the Llama model : 

<ol> 
<li> Create a folder to store the llama-model -> <code>mkdir llama-model</code></li>
<li> Download the Llama3.2-3B model from : <a href = https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main?show_file_info=Llama-3.2-3B-Instruct-Q4_0_4_4.gguf> here </a> </li>
<li> Move the model into the llama-model folder </li>
</ol>

## Getting the whisper model:
<ol>
<li> Create a folder to store the whisper-model -> <code>mkdir whisper-model</code> </li>
<li> Download <a href = https://huggingface.co/danielus/ggml-whisper-models/tree/main>ggml-tiny-fp16.bin</a></li>
<li>Move the model into whisper-model</li>
<li>Quantize the model to 4 bit (if necessary) using the following command :<code> whisper.cpp/quantize whisper-model/ggml-tiny-fp16.bin whisper-model/ggml-tiny-q4_0.bin q4_0</code> </li>
<li> Delete the fp16 model (if Q4 is being used) to save space </li>
</ol>

## Setup the nyuntam code base :
The code is present in nyunta, so we need to get that. 

<ol>
<li><code>git clone https://github.com/nyunAI/nyuntam.git</code> </li>
<li> <code>cd nyuntam </code></li>
<li> The code is currently in the "tts" branch : <code>git checkout origin/tts</code> </li>
</ol>

## Running the code : 
<ol>
<li> Move into the appropriate folder : <code>nyuntam/examples/experimentals/voice-engine</code> </li>
<li> Put the correct executable (llama-server, whisper-server) path for your system in the yaml file present in <code>recipe/recipe_android.yaml</code> . **NOTE :** These servers are present inside llama.cpp and whisper.cpp respectively </li>
<li> Put the correct model file path in the recipe yaml file </li>
<li>run the main_android.py using the following command : <code>python3.11 main_android.py --config recipe/recipe_android.yaml</code> </li>
</ol>

**NOTE :** If you are running this for the first time, there maybe a few packages that will be missing. These can be installed using  `pip3.11 install [package-name]`



