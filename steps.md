1. Activate venv

# windows powershell
.venv\Scripts\Activate.ps1
# linux/mac
source .venv/bin/activate


2. Preprocess raw .psv files into per-patient .pt (parallel).

python src/parallel_preprocess.py --raw_folder data/raw --out_folder data/processed/patients --seq_len 48 --nprocs 8


(Optional) pack into LMDB for faster IO (recommended for 40k files).

python src/lmdb_packer.py --in_folder data/processed/patients --out_folder data/processed/lmdb_shards --shard_size 4000
# use LMDB index:
cp data/processed/lmdb_shards/index_lmdb.pt data/processed/patients/index_with_labels_lmdb.pt


3. Train local baseline (Transformer):

python src/train_local.py --index data/processed/patients/index_with_labels.pt --epochs 10 --batch_size 64 --model transformer


4. Train GRU-D (missing-data aware):

python src/train_local.py --index data/processed/patients/index_with_labels.pt --epochs 10 --batch_size 64 --model grud


5. Hyperparameter quick grid (small):

python src/hyperparam_search.py


Federated simulation:

6. split clients (3–5)

python src/split_clients.py --processed_folder data/processed/patients --out_root data/processed/clients --n_clients 3


7. start server

# (Terminal 1)
python src/fl_server.py --model transformer --n_features 40 --seq_len 48 --min_clients 2 --rounds 5


8. start each client (terminal 2/3/4)

# (Terminal 2)
python src/fl_client.py --index data/processed/clients/client1/index.pt --model transformer --server_address 127.0.0.1:8080

# (Terminal 3)
python src/fl_client.py --index data/processed/clients/client2/index.pt --model transformer --server_address 127.0.0.1:8080

# (Terminal 4)
python src/fl_client.py --index data/processed/clients/client3/index.pt --model transformer --server_address 127.0.0.1:8080


9. Secure aggregation PoC (run locally to show the masks cancel):

python src/secure_agg_poc.py


10. Train the "Local-Only" Baseline Model

In a terminal, run train_local.py to train a model only on client1's data.

python src/train_local.py --index data/processed/clients/client1/index.pt --model transformer --run_name local_client1_model --epochs 10

When it finishes, it will print a Run folder:. Copy the path to the model_best.pt file inside, which will look something like: runs\20251115T123000Z__local_client1_model\checkpoints\model_best.pt

11. Evaluate the Federated Model

Run evaluate.py to test your federated model against the client3 test set.

python src/evaluate.py --index data/processed/clients/client3/index.pt --ckpt server_out/global_best.pt --model transformer --n_features 40 --seq_len 48 --out_file eval_results_federated.json

12. Evaluate the Local-Only Model

Run evaluate.py again, but this time using your local model.

(Remember to paste the full path from step 4.1)

python src/evaluate.py --index data/processed/clients/client3/index.pt --ckpt "runs\20251115T123000Z__local_client1_model\checkpoints\model_best.pt" --model transformer --n_features 40 --seq_len 48 --out_file eval_results_local.json

13. Generate the Final Plot

Run plot_results.py to combine your two evaluation files into one comparison graph.

python src/plot_results.py --results eval_results_federated.json eval_results_local.json --out_file model_comparison_plot.png