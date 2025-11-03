# Jetson-Nano
Various AI projects for my Jetson Nano

I was lent a Dell Pro Max DG10 development system awhile back. I had it for about a week and it was a facinating piece of equipment. Unfortunately I did have to give it back, but the experience of having a machine like that strictly to develope AI applications was kind of fun. Also unfortunately, those things cost $4,000 which is outside of my budget. Fortunately, NVIDIA makes or rather made, smaller more affordable systems that serve the same purpose. Of course the Jetson Thor's and such were also out of my price range, but the Jetson Orin Nano came in at $250, not bad all things considered. Sure its not as powerful and leaves a lot of things to be desired, but for a little hobbyist toy, it works just fine for me.

This little machine only has 8 GB of RAM in total, this RAM is shared by both the CPU and the GPU. Hyopthetically, you should be able to use a 6 or 7 billion parameter model, however, in reailty, the overhead of the host operating system means this is much less. I have found about the biggest you can go is 4 billion (4b). The LLM used in Simple-chat is a 4b model and it is almost too much. The RAG chatbot is much smaller, 1.5b in fact. It had to be smaller because the RAG web frontend is much more complex, it also uses a secondary model. Both of these, when up and running, consume about the same amount of RAM. Of Course you could never run them at the same time on the Nano.

My goal with each of these projects is for each of them to be self contained and easy to deploy and reproducible, hence the build shell scripts. Everything should be in its own container, so as not to mess with anything other models and especially the host system. Hypathetically you should be able to build these on any Linux machine with Docker installed and a GPU with 8GB od VRAM. I have not tested them on anything other than my Nano, so milage may vary.

All of these applications should be accessble at port 8080, so you can point a web browser at the Nano as such, http://192.168.0.200:8080 and the web interface will come up. If you are deploying them all at once running a on beefier system, you will need to change the ports on a couple of them. Since I only run them one at a time, putting them all on the same port is convient.

They all use Flask for the web interface, I chose it because it consumes 200-300 MB less RAM than Gradio. Flask is also much more flexable and customizable, making it a better choice for changing and building off of.

View these projects as starting points rather than finished products. There are plenty of possile changes and upgrades that would make these programs more robust or change what they do to better suit your purpose.

As a side note, only llm-chat uses an uncensored model, the others use standard models with safe guards in place.

----------

Installation

> sudo apt install docker-compose docker.io

> sudo usermod -aG docker $USER

> curl -fsSL https://ollama.com/install.sh | sh

> git clone https://github.com/dusty-nv/jetson-containers

> bash jetson-containers/install.sh

> git clone https://github.com/cjstoddard/Jetson-Nano.git

> cd Jetson-Nano

Choose which container you want to build, cd into the folder and run build.sh

> cd llm-chat

> chmod +x build.sh

> ./build.sh

----------

Troubleshooting

If any of these break for no apparent reason, the first thing I would do is re-run the build script and say yes when it asks if you want a clean build. If that does not work, then something in your enviroment has changed, maybe run "docker compose logs -f" to see what is happening.

The command "docker compose logs -f" is your best friend, if there is something wrong, it will probably be really obvious what it is by dumping the logs like this. You can copy and paste the output into ChatGPT or Claude and they will analyse it for you. If you ask me for help, the first thing I will do is ask you for the output of this command so I can have Claude or ChatGPT analyse it for me, so you might as well cut out the middle man and do it yourself.

If you are still  having trouble geting one of these to work, chances are good you are simply running out of RAM. All of these run close to the edge, even if it does not appear so. Make sure you shut down containers with "docker compose down" after you are finished using it. Second, install jtop on your system, this is a nice monitoring tool that will give you a great deal of information about what is happening on the device.

> sudo apt update

> sudo apt install python3-pip

> sudo pip3 install -U jetson-stats

> jtop

Once open, if you press 4, it will give you specific information on memory usage, more importantly, on this screen if you press c, it will clear the memory cache. Do not do this while a container is running, it might end badly. Alternatively, you can just run this command;

> sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

Clearing the cache between using these programs will free a couple hundred MB of RAM and is a good best practice.

If you are still having trouble, disbale the zram and use a swap file. I suggest this as zram uses actual RAM and disabling it frees up a couple of hundred megabytes of RAM. By using a swap file, you will instead be using the storage for swap. Keep in mind this adds wear and tear to you storage and will slow responses down on larger models.

> sudo systemctl disable nvzramconfig

> sudo dd if=/dev/zero of=/swapfile bs=1M count=16384

> sudo chmod 600 /swapfile

> sudo mkswap /swapfile

> sudo swapon /swapfile

> sudo nano /etc/fstab

Add this line to the end;

> /swapfile none swap sw 0 0

Save and exit, then reboot the system.

----------

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

