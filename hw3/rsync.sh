#!/usr/bin/env bash

rsync -avz --exclude "__pycache__/" --exclude "data/" . $1:/home/herizhao/cs294/hw3