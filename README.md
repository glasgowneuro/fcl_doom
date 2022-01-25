# FCL DOOM

This is a demo where FCL fights against an Intel agent from the
vizdoom competition.

## Prequisites

```
apt install docker.io
systemctl start docker
systemctl enable docker
```

Edit `/etc/group` and add your username to the group "docker".

## Building

Build these two dockers:
```
./build.sh intelact
./build.sh host
```

Clone vizdoom with:
```
git clone https://github.com/mwydmuch/ViZDoom.git
```

Install it with:
```
sudo pip3 install .
```

## Running the demo

First run the host, followed by the bots:
```
./run.sh host
./run.sh intelact
```
The best approach is to run this in two separate terminal windows.

Our own FCL agent runs just in the Linux system itself and connects
to the host.

To run the DFL bot:
```
python3 ./run_agentFCL.py 0.0001
```
where 0.0001 is the learning rate.

Which runs a deathmatch for the length of the time specified 
in `host/host.py`, and will write to 3 files:
```
KD.txt
wtDist.txt
FCLOutput.txt
```

The first logs a timestamp each time the bot is killed.
To calculate a KD ratio, you will need to edit
`intelact/IntealAct_track2/run_agent.py` to write
the corresponding death events for the intel bot.
`wtDist.txt` logs the average Euclidean distance
of the weights from their initial values, per layer.
`FCLOutput.txt` logs various measures:
feedback error, bot steering action, health. See the code. 


If you wish to experiment further:
----------------------------------
There are a number of *.cfg files here - if you want to run deathmatch in the way described above, copy _vizdoomNetwork.cfg to _vizdoom.cfg. Assuming the host is already running, running run_agentDFL.py will then attempt to connect and start a match. It's crucial that all bots, and the host, are using the same Doom map. This is specified in their own .cfg files.  

You can use any of the .wad files that come with VizDoom, but we recommend editing them to a) have more spawn points so that the bot gets a better randomisation of starting position and increase generalisation of learning, and b) to increase the amount of ammo available, otherwise the enemy can run out of ammo and start behaving strangely. We used http://www.doombuilder.com/ for this. 

If you want to change the actions available to the bot, you need to edit config/config.cfg. At the moment, the bot only rotates in the plane, and shoots, so there is lots of scope for improvement!

The intelact robot uses the CPU for its machine learning which in turn uses tensorflow. Feel free to install the tensorflow library which GPU support instead.
