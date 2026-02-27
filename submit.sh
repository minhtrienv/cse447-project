#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Heon Jwa,heonjwa\nTrien Vuong,mtrienv\nJesse shieh, shiehj3" > submit/team.txt

# train model
python3 -u src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
python3 -u src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file and requirements
cp Dockerfile submit/Dockerfile
cp requirements.txt submit/requirements.txt

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
