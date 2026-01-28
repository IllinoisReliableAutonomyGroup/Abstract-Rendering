#!/bin/bash

# Check if case_name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <case_name> [skip (optional)]"
    exit 1
fi

case_name=$1
skip_certification=${2:-""}  # Default to an empty string if not provided

cd /home/chenxij2/Abstract-Rendering-CAV
echo "Starting certification and plotting for case: ${case_name}"

if [ "$skip_certification" != "skip" ]; then
    # Run certify_gatenet.py
    echo "Running certification..."
    python3 scripts/certify_gatenet.py --config configs/${case_name}/gatenet.yml
    if [ $? -ne 0 ]; then
        echo "Certification failed for case: ${case_name}"
        exit 1
    fi
    echo "Certification completed successfully for case: ${case_name}"
    echo ""
else
    echo "Skipping certification as requested."
fi

# Run plot_gatenet.py
echo "Generating plots..."
python3 scripts/plot_gatenet.py --config configs/${case_name}/gatenet.yml --traj configs/${case_name}/traj.yaml
if [ $? -ne 0 ]; then
    echo "Plot generation failed for case: ${case_name}"
    exit 1
fi
echo "Plot generation completed successfully for case: ${case_name}"

echo "Certification and plotting completed for case: ${case_name}"