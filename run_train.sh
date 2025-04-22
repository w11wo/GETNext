for city in beijing istanbul jakarta kuwait_city melbourne moscow new_york petaling_jaya sao_paulo shanghai sydney tokyo; do
    python build_graph.py \
        --csv_path data/$city/${city}_checkins_train.csv \
        --output_dir data/$city

    python train.py \
        --data-train data/$city/${city}_checkins_train.csv \
        --data-val data/$city/${city}_checkins_test.csv \
        --data-adj-mtx data/$city/graph_A.csv \
        --data-node-feats data/$city/graph_X.csv \
        --time-units 48 --timestamp_column timestamp \
        --poi-embed-dim 128 --user-embed-dim 128 \
        --time-embed-dim 32 --cat-embed-dim 32 \
        --node-attn-nhid 128 \
        --transformer-nhid 1024 \
        --transformer-nlayers 2 --transformer-nhead 2 \
        --batch 16 --epochs 200 --name $city
done