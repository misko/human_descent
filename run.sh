#!/bin/bash
echo "Download MNIST"
python hudes/websocket_server.py --run-in thread --download-dataset-and-exit
echo "Done downloading MNIST"
python hudes/websocket_server.py --device mps --run-in thread --port 8765 &
server_pid=$!
python hudes/hudes_play.py --input keyboardGL
python hudes/hudes_play.py --input keyboard
#python hudes/hudes_play.py --input xtouch
kill ${server_pid}
kill -9 ${server_pid}
# /home/mouse9911/gits/human_descent/hudes_env/bin/python websocket_server.py --device=cuda --model=cnn3 --port=10000 --ssl-pem=hudes.pem --run-in=process
