#!/bin/bash

# Run training 5 times sequentially
for f in $(seq 0.6 0.1 0.9)
do
     # Replace comma with period if needed
    f=$(echo $f | sed 's/,/./')

    echo "🔁 Starting testing fraction #$f with severity #$i..."
        python3 toy_experiments/evaluate_ensemble_ood.py --ood_fraction $f --ood_dataset svhn --ensemble_type deep --num_models 5 --data_transformation transf_data
        echo "✅ Finished testing fraction #$f with severity #$i."
done

echo "🎉 All evaluation runs completed."