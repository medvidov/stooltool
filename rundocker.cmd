@echo off
setlocal

if "%CI_JOB_TOKEN%"=="" goto :err_set_token

set PIP_EXTRA_INDEX_URL=https://__token__:%CI_JOB_TOKEN%@gitlab.com/api/v4/projects/20511699/packages/pypi/simple
docker build --build-arg PIP_EXTRA_INDEX_URL=%PIP_EXTRA_INDEX_URL% -f Dockerfile.gunicorn --tag liverfast-local .
if errorlevel 1 goto :eof

docker run -p 5000:5000 -it --rm liverfast-local 
goto :eof

:err_set_token
echo You must have an environment variable named CI_JOB_TOKEN with a personal access token assigned to it.
echo.
echo To create a personal access token, log into gitlab.com and select your account settings.  From there,
echo select Access Tokens and generate a personal access token.  You will need to have read_api, read_repository
echo and read_registry permissions.
echo.
goto :eof

