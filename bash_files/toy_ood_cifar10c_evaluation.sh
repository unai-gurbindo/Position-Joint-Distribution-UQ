#!/bin/bash

# Run training 5 times sequentially
for f in $(seq 0.1 0.1 0.4)
do
     # Replace comma with period if needed
    f=$(echo $f | sed 's/,/./')

    for i in {1..5}
    do
        echo "🔁 Starting testing fraction #$f with severity #$i..."
        python3 toy_experiments/evaluate_ensemble_ood.py --ood_fraction $f --ood_dataset cifar10c --ensemble_type deep --cifar10c_severity $i --num_models 5 --data_transformation transf_data
        echo "✅ Finished testing fraction #$f with severity #$i."
    done
done

echo "🎉 All evaluation runs completed."