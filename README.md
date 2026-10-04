# ConnPOMO

ConnPOMO is the deep reinforcement learning route constructor of the paper "Connectivity-Aware Deep Reinforcement Learning for Rideable Multi-Stop Itineraries on City Bus Networks".

- Install: `pip install -r requirements.txt`
- Train the 13 models: `python training/run_all.py grid --epochs 50 && python training/run_all.py all --epochs 100 --min-epochs 50 --patience 30 --val-inst 256`
- Plan a route: `python -m connpomo.evaluate --N 30 --density hard`, or `--instances file.npz` with stacked arrays `xy` (from `connpomo.data.unit_square(latlon)`), `conn` and `dist` (haversine km); the checkpoint rule picks the model unless `--ckpt` is given
- Synthetic benchmark (260 instances): `python -m connpomo.data`
- easy, medium and hard are the dense, medium and sparse classes of the paper.
- The real Astana and Shymkent instances are available from the corresponding author on request.
