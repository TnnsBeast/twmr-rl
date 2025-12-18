# Transformable Wheel Robot

## Git Workflow

### Intiail Setup

```bash
# This only needs to be done once and you've likely done it already
git config --global user.name "YOUR NAME"
git config --global user.email "YOU@EXAMPLE.COM"

# Clone the repository
git clone https://github.com/JacobBau04/Transformable-Leg-Wheel-Robot
```

### Work cycle

#### For Smaller Changes

For smaller changes you can commit directly to main.

```bash
# Make changes to the code and then (repeat as is useful)
git add FILES_YOU_CHANGED/ADDED
git commit -m "SHORT DESCRIPTIVE MESSAGE"

# Pull latest changes from main (this will fetch and merge)
git pull origin main

# Cleanup any merge conflicts and then push changes to main
git push origin main
```

#### For Larger Changes

I recommend creating a branch when you are working on a large change (a new feature, a bug fix, etc).
This keeps your changes isolated from the main branch that we all use until they are well tested and ready to be merged in.

```bash
# Create and switch to a new branch named
git checkout -b feat/SHORT-DESCRIPTIVE-NAME

# Make changes to the code and then (repeat as is useful)
git add FILES_YOU_CHANGED/ADDED
git commit -m "SHORT DESCRIPTIVE MESSAGE"

# Push branch to remote repository
git push -u origin feat/SHORT-DESCRIPTIVE-NAME

# Continue working on the feature branch until ready to merge
# Open a Pull Request (PR) on GitHub targeting main and then send a message to have it reviewed
```
