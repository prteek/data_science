#!/bin/bash

git clone -b $1 --single-branch https://$GITHUB_USER:$GITHUB_TOKEN@github.com/PropertyLift/ds.bricklane.git /usr/src/app/repo
cd /usr/src/app/repo/analytics
git checkout $2
dbt run -m $3
dbt test -m $3