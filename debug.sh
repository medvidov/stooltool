#!/bin/bash
set -e

if [ -z ${CI_JOB_TOKEN} ]; then
    echo You must have an environment variable named CI_JOB_TOKEN with a personal access token assigned to it.
    echo ""
    echo To create a personal access token, log into gitlab.com and select your account settings.  From there,
    echo select Access Tokens and generate a personal access token.  You will need to have read_api, read_repository
    echo and read_registry permissions.
    echo ""
else
    export PIP_EXTRA_INDEX_URL=https://__token__:${CI_JOB_TOKEN}@gitlab.com/api/v4/projects/20511699/packages/pypi/simple
    pip install .

    export FLASK_ENV=development
    export FLASK_APP=app
    flask run
fi

