if [ -z $1 ] ; then
	echo  "Please provide the device to run the experiments."
	exit 1
fi

mkdir -p /root/workspace/out/synth_chains

# -------- multi classes ------------
for seed in 0 1 2 3 4 5 6 7 8 9
do  
    # # APPNP
    # # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1appnp-memory-timing-A\2/" /root/workspace/PR-inspired-aggregation/baselines/appnp/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/appnp/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/appnp-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.01 \
    # train.wd=0.00001 \
    # model.hidden_channels=16 \
    # model.hidden_layers=1 \
    # model.dropout=0.8 \
    # model.alpha=1.0 \
    # model.K=200 
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1appnp\2/" /root/workspace/PR-inspired-aggregation/tasks/synth_chains.py

    # # GCN
    # # ---
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1gcn-memory-timing-A\2/" /root/workspace/PR-inspired-aggregation/baselines/gcn/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/gcn/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/gcn-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.01 \
    # train.wd=0.001 \
    # model.dropout=0.0 \
    # model.hidden_channels=16 \
    # model.hidden_layers=150
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1gcn\2/" /root/workspace/PR-inspired-aggregation/baselines/gcn/synth_chains.py

    # # GCNII
    # # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1gcnii-memory-timing-A\2/" /root/workspace/PR-inspired-aggregation/baselines/gcnii/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/gcnii/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/gcnii-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.01 \
    # train.wd=0.00001 \
    # model.hidden_channels=16 \
    # model.hidden_layers=170 \
    # model.alpha=0.0 \
    # model.theta=0.0 \
    # model.dropout=0.0
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1gcnii\2/" /root/workspace/PR-inspired-aggregation/baselines/gcnii/synth_chains.py

    # # GPRGNN
    # # -------
    sed -i "s/\(.*name='\)[^']*\('.*\)/\1gprgnn-memory-timing-A\2/" /root/workspace/PR-inspired-aggregation/baselines/gprgnn/synth_chains.py
    CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/gprgnn/synth_chains.py \
    setup.seed=$seed \
    setup.device=cuda \
    setup.sweep=True \
    data.chain_len=100 \
    data.num_chains=20 \
    data.num_classes=10 \
    data.feature_dim=100 \
    data.noise=0.0 \
    load.split=fixed_05/10/85 \
    load.checkpoint_path=/root/workspace/out/synth_chains/gprgnn-memory-timing.pt \
    train.epochs=2000 \
    train.patience=100 \
    train.lr=0.01 \
    train.wd=0.000000000001 \
    model.hidden_channels=16 \
    model.hidden_layers=1 \
    model.dropout=0.3 \
    model.gamma=0.8 \
    model.alpha=0.8 \
    model.K=200
    sed -i "s/\(.*name='\)[^']*\('.*\)/\1gprgnn\2/" /root/workspace/PR-inspired-aggregation/baselines/gprgnn/synth_chains.py

    # EIGNN
    # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1eignn-memory-timing-B\2/" /root/workspace/PR-inspired-aggregation/baselines/eignn/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/eignn/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/eignn-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.01 \
    # train.wd=0.0 \
    # model.hidden_channels=16 \
    # model.dropout=0.5 \
    # model.num_eigenvec=100 \
    # model.gamma=0.8
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1eignn\2/" /root/workspace/PR-inspired-aggregation/baselines/eignn/synth_chains.py

    # IGNN
    # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1ignn-memory-timing-B\2/" /root/workspace/PR-inspired-aggregation/baselines/ignn/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/ignn/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/ignn-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.01 \
    # train.wd=0.000001 \
    # model.hidden_channels=16 \
    # model.dropout=0.5 \
    # model.kappa=0.95 \
    # model.max_iter=300
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1ignn\2/" /root/workspace/PR-inspired-aggregation/baselines/ignn/synth_chains.py

    # IGNN(Init)
    # -----
    sed -i "s/\(.*name='\)[^']*\('.*\)/\1ignn-init-memory-timing-B\2/" /root/workspace/PR-inspired-aggregation/baselines/ignn/synth_chains.py
    CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/ignn/synth_chains.py \
    setup.seed=$seed \
    setup.device=cuda \
    setup.sweep=True \
    data.chain_len=100 \
    data.num_chains=20 \
    data.num_classes=10 \
    data.feature_dim=100 \
    data.noise=0.0 \
    load.split=fixed_05/10/85 \
    load.checkpoint_path=/root/workspace/out/synth_chains/ignn-memory-timing.pt \
    train.epochs=2000 \
    train.patience=100 \
    train.lr=0.01 \
    train.wd=0.000001 \
    model.hidden_channels=16 \
    model.dropout=0.5 \
    model.kappa=0.95 \
    model.max_iter=300 \
    model.reuse_fp=True
    sed -i "s/\(.*name='\)[^']*\('.*\)/\1ignn\2/" /root/workspace/PR-inspired-aggregation/baselines/ignn/synth_chains.py

    # # PRGNN
    # # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1prgnn-memory-timing-D\2/" /root/workspace/PR-inspired-aggregation/tasks/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/tasks/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/prgnn-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.009 \
    # train.wd=0.0\
    # model.hidden_channels=16 \
    # model.dropout=0.0 \
    # model.phantom_grad=2 \
    # model.beta_init=0.9 \
    # model.gamma_init=0.0 \
    # model.max_iter=10 \
    # model.tol=0.000003
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1prgnn\2/" /root/workspace/PR-inspired-aggregation/tasks/synth_chains.py

    # # PPNP
    # # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1ppnp-memory-timing-A\2/" /root/workspace/PR-inspired-aggregation/baselines/ppnp/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/ppnp/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/ppnp-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.01 \
    # train.wd=0.00000001 \
    # model.hidden_channels=16 \
    # model.hidden_layers=1 \
    # model.dropout=0.2 \
    # model.alpha=0.0 \
    # model.K=200
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1ppnp\2/" /root/workspace/PR-inspired-aggregation/baselines/ppnp/synth_chains.py

    # # SGC
    # # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1sgc-memory-timing-A\2/" /root/workspace/PR-inspired-aggregation/baselines/sgc/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/sgc/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/sgc-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.009 \
    # train.wd=0.00000001 \
    # model.hidden_channels=16 \
    # model.dropout=0.8 \
    # model.hidden_layers=1 \
    # model.K=190
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1sgc\2/" /root/workspace/PR-inspired-aggregation/baselines/sgc/synth_chains.py

    # # SSGC
    # # -----
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1ssgc-memory-timing-A\2/" /root/workspace/PR-inspired-aggregation/baselines/ssgc/synth_chains.py
    # CUDA_VISIBLE_DEVICES=$1 python3 /root/workspace/PR-inspired-aggregation/baselines/ssgc/synth_chains.py \
    # setup.seed=$seed \
    # setup.device=cuda \
    # setup.sweep=True \
    # data.chain_len=100 \
    # data.num_chains=20 \
    # data.num_classes=10 \
    # data.feature_dim=100 \
    # data.noise=0.0 \
    # load.split=fixed_05/10/85 \
    # load.checkpoint_path=/root/workspace/out/synth_chains/ssgc-memory-timing.pt \
    # train.epochs=2000 \
    # train.patience=100 \
    # train.lr=0.009 \
    # train.wd=0.00000001 \
    # model.hidden_channels=16 \
    # model.dropout=0.3 \
    # model.hidden_layers=1 \
    # model.alpha=0.8 \
    # model.K=200
    # sed -i "s/\(.*name='\)[^']*\('.*\)/\1ssgc\2/" /root/workspace/PR-inspired-aggregation/baselines/ssgc/synth_chains.py
done