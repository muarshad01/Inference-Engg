#!/bin/bash

clear 

# ssh-add -k ~/.ssh/id_rsa

git pull

git add .
git commit -m "update"
git push
