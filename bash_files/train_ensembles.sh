#!/bin/bash

# Run training 5 times sequentially
for i in {0..4}
    do
        echo "🔁 Starting training ensemble #$i..."
        python3 toy_experiments/train_ensemble.py --seed $i --ensemble_type deep --num_models 10
        echo "✅ Finished training ensemble #$i."
    done

echo "🎉 All training runs completed."