#/bin/bash

docker build -t tp-codage-container .
docker run -it --rm --name tp-codage -v "$PWD":/usr/src/myapp -w /usr/src/myapp tp-codage-container #python your-daemon-or-script.py