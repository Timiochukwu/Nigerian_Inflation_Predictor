#!/bin/bash
# Check if all required data files are present

echo "🔍 Checking for required data files..."
echo ""

FILES=(
    "credit-risk-api/data/raw/traindemographics.csv"
    "credit-risk-api/data/raw/trainperf.csv"
    "credit-risk-api/data/raw/trainprevloans.csv"
)

all_present=true

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "✅ Found: $(basename $file) (${size})"
    else
        echo "❌ Missing: $(basename $file)"
        all_present=false
    fi
done

echo ""
if [ "$all_present" = true ]; then
    echo "🎉 All files present! You're ready to start Day 1!"
    echo ""
    echo "Next steps:"
    echo "1. cd credit-risk-api"
    echo "2. pip install -r requirements_day1.txt"
    echo "3. jupyter notebook"
    echo "4. Create notebooks/day1_exploration.ipynb"
else
    echo "⚠️  Please upload the missing files to credit-risk-api/data/raw/"
fi
