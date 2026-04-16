#!/bin/bash
# Push script - run this when GitHub credentials are available

echo "Pushing to https://github.com/Krishal743/Cascade-rot.git"

cd "$(dirname "$0")"

# Check if origin remote exists
if git remote get-url origin &>/dev/null; then
    echo "Remote 'origin' exists"
else
    echo "Adding remote..."
    git remote add origin https://github.com/Krishal743/Cascade-rot.git
fi

# Push
echo "Pushing to master..."
git push -u origin master

echo "Done!"
