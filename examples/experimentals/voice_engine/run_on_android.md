## Stes to run this on android device : 

<ol> 
<li> Install termux apk : <a href=https://github.com/termux/termux-app/releases/tag/v0.118.1> link </a>  </li>

<li> Termux is not working on latest version of android, so, preferable use android 9 (tested) </li>

<li> in termux, run : apt upgrade && apt update </li>
<li> install python in termux : <ol> <li>pkg install tur-repo</li> <li>pkg install python3.11</li></ol>
<li>clone and build llama.cpp using make </li>
<li> download llama models and whisper models and save it in the paths given in the recipe_android.yaml </li>
<li> use pip3.11 to downlaod the required pakcages </li>
<li> Install espeak using :  pkg install espeak </li>
<li> run the main_android.py using recipe_android.yaml </li>
</ol>


