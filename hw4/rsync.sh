#!/usr/bin/env bash

rsync -avz --exclude "__pycache__/" --exclude "data/" --exclude "gcloud/" --exclude "report/" . $1:/home/herizhao/cs294/hw4